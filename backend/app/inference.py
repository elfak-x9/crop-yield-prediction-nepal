"""
Live inference engine.

This mirrors the logic in src/preprocessing.py and src/scaling.py, but
builds ONE sample at a time (given a district + year) instead of scanning
the full yield_df like training does. Soil data and climate data are
loaded once and cached in memory; models and scalers are loaded lazily
per crop and cached after first use.
"""

import threading
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config

_lock = threading.Lock()
_model_cache = {}
_scaler_cache = {}

# Target sequence length expected by the CNN-LSTM model (12 months per year)
MODEL_SEQUENCE_LENGTH = 12


class PredictionError(Exception):
    """Raised for any user-facing prediction failure (bad district, no data, etc.)."""


# ---------------------------------------------------------------------------
# Cached raw data loading (soil.csv + climate.csv only)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_soil_df():
    df = pd.read_csv(config.SOIL_CSV)
    df.columns = df.columns.str.strip()
    df["District"] = df["District"].astype(str).str.strip().str.lower()
    return df


@lru_cache(maxsize=1)
def _load_climate_df():
    df = pd.read_csv(config.CLIMATE_CSV)
    df.columns = df.columns.str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Year"] = df["DATE"].dt.year.astype(int)
    return df


@lru_cache(maxsize=1)
def _district_to_coord():
    """Same nearest-grid-point matching used in preprocessing.py."""
    soil_df = _load_soil_df()
    climate_df = _load_climate_df()
    climate_coords = climate_df[["LAT", "LON"]].drop_duplicates().values

    mapping = {}
    for _, row in soil_df.iterrows():
        dist = row["District"]
        s_lat, s_lon = float(row["lat"]), float(row["lon"])
        distances = np.sqrt(
            (climate_coords[:, 0] - s_lat) ** 2 + (climate_coords[:, 1] - s_lon) ** 2
        )
        closest_idx = np.argmin(distances)
        mapping[dist] = (climate_coords[closest_idx][0], climate_coords[closest_idx][1])
    return mapping


@lru_cache(maxsize=1)
def _soil_vectors():
    soil_df = _load_soil_df()
    soil_raw = soil_df[config.SOIL_COLS].values
    soil_dummies = pd.get_dummies(soil_df["parent_soil"], prefix="soil_type", dtype=float).values

    vectors = {}
    for idx, row in soil_df.iterrows():
        district = row["District"]
        vectors[district] = np.hstack([soil_raw[idx], soil_dummies[idx]]).astype(np.float32)
    return vectors


def list_districts():
    return sorted(_load_soil_df()["District"].unique().tolist())


def list_years_for_district(district: str):
    district = district.strip().lower()
    coord_map = _district_to_coord()
    if district not in coord_map:
        raise PredictionError(f"Unknown district: '{district}'")
    c_lat, c_lon = coord_map[district]
    climate_df = _load_climate_df()
    sub = climate_df[np.isclose(climate_df["LAT"], c_lat) & np.isclose(climate_df["LON"], c_lon)]
    return sorted(sub["Year"].unique().tolist())


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
def _build_climate_sequence(district: str, year: int) -> np.ndarray:
    coord_map = _district_to_coord()
    if district not in coord_map:
        raise PredictionError(f"Unknown district: '{district}'")
    c_lat, c_lon = coord_map[district]

    climate_df = _load_climate_df()
    sub_climate = climate_df[
        np.isclose(climate_df["LAT"], c_lat)
        & np.isclose(climate_df["LON"], c_lon)
        & (climate_df["Year"] == year)
    ].sort_values("DATE")

    if len(sub_climate) == 0:
        raise PredictionError(f"No climate data for district '{district}' in year {year}")

    seq = sub_climate[config.CLIMATE_FEATURES].values

    # Enforce exactly 12 timesteps to match the Keras model shape (12, 6)
    if len(seq) >= MODEL_SEQUENCE_LENGTH:
        seq = seq[:MODEL_SEQUENCE_LENGTH]
    else:
        pad_width = MODEL_SEQUENCE_LENGTH - len(seq)
        padding = np.tile(seq[-1:], (pad_width, 1))
        seq = np.vstack([seq, padding])

    return seq.astype(np.float32)


def _build_soil_vector(district: str) -> np.ndarray:
    vectors = _soil_vectors()
    if district not in vectors:
        raise PredictionError(f"Unknown district: '{district}'")
    return vectors[district]


# ---------------------------------------------------------------------------
# Model + scaler loading
# ---------------------------------------------------------------------------
def _get_model(crop: str):
    if crop not in _model_cache:
        with _lock:
            if crop not in _model_cache:
                path = f"{config.SAVE_DIR}/best_{crop}_model.keras"
                try:
                    _model_cache[crop] = tf.keras.models.load_model(path)
                except (OSError, IOError) as exc:
                    raise PredictionError(
                        f"Model file not found for crop '{crop}' at {path}"
                    ) from exc
    return _model_cache[crop]


def _get_scalers(crop: str):
    if crop not in _scaler_cache:
        with _lock:
            if crop not in _scaler_cache:
                try:
                    scaler_climate = joblib.load(f"{config.SAVE_DIR}/{crop}_scaler_climate.pkl")
                    scaler_soil = joblib.load(f"{config.SAVE_DIR}/{crop}_scaler_soil.pkl")
                    scaler_y = joblib.load(f"{config.SAVE_DIR}/{crop}_scaler_y.pkl")
                except (OSError, IOError) as exc:
                    raise PredictionError(
                        f"Scaler files not found for crop '{crop}' in {config.SAVE_DIR}"
                    ) from exc
                _scaler_cache[crop] = (scaler_climate, scaler_soil, scaler_y)
    return _scaler_cache[crop]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def predict_yield(crop: str, district: str, year: int) -> float:
    """
    Returns the predicted yield in mt/ha for one crop/district/year.
    """
    if crop not in config.CROPS:
        raise PredictionError(f"Unsupported crop code: '{crop}'")

    district = district.strip().lower()

    climate_seq = _build_climate_sequence(district, year)  # Shape: (12, 6)
    soil_vec = _build_soil_vector(district)

    scaler_climate, scaler_soil, scaler_y = _get_scalers(crop)

    # Climate: flatten to 2D to scale, then reshape to (1, 12, 6)
    seq_len, n_climate_feats = climate_seq.shape
    clim_flat = climate_seq.reshape(-1, n_climate_feats)
    clim_scaled = scaler_climate.transform(clim_flat).reshape(1, seq_len, n_climate_feats)

    soil_scaled = scaler_soil.transform(soil_vec.reshape(1, -1))

    model = _get_model(crop)
    pred_scaled = model.predict(
        {"Climate_Input": clim_scaled, "Soil_Input": soil_scaled}, verbose=0
    ).flatten()

    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    return float(pred)