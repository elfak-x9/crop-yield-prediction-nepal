"""
Live inference engine.

This mirrors the logic in src/preprocessing.py and src/scaling.py, but
builds ONE sample at a time (given a district + year) instead of scanning
the full yield_df like training does. Soil data and climate data are
loaded once and cached in memory; models and scalers are loaded lazily
per crop and cached after first use.
"""

import io
import threading
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import config

_lock = threading.Lock()
_model_cache = {}
_scaler_cache = {}
_stats_cache = {}

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


def latest_year(district: str) -> int:
    """Most recent year with climate data for a district (used as default)."""
    years = list_years_for_district(district)
    if not years:
        raise PredictionError(f"No climate data available for district '{district}'")
    return int(years[-1])


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


def predict_with_confidence(crop: str, district: str, year: int) -> dict:
    """
    Runs a prediction and attaches model confidence info derived from the
    crop's held-out validation metrics (same split used by evaluate.py).
    """
    yield_mt_ha = predict_yield(crop, district, year)
    stats = compute_crop_stats(crop)

    mae = stats["mae_mt_per_ha"]
    mean_y = stats["mean_actual_mt_per_ha"]

    # Confidence: how close the typical error is to the typical yield,
    # expressed as a percentage. Clamped to a sane range.
    if mean_y > 0:
        confidence_pct = round(max(0.0, min(0.999, 1.0 - mae / mean_y)) * 100, 1)
    else:
        confidence_pct = 0.0

    return {
        "yield_mt_per_ha": yield_mt_ha,
        "confidence_pct": confidence_pct,
        "error_margin_mt_per_ha": round(mae, 4),
    }


# ---------------------------------------------------------------------------
# Model statistics (computed once per crop on the held-out validation split)
# ---------------------------------------------------------------------------
def compute_crop_stats(crop: str) -> dict:
    """
    Evaluate a crop's model on the same 20% validation split that evaluate.py
    uses and return R², RMSE, MAE, sample counts and a text summary of the
    model architecture. Results are cached after the first call.
    """
    if crop not in config.CROPS:
        raise PredictionError(f"Unsupported crop code: '{crop}'")

    with _lock:
        if crop in _stats_cache:
            return _stats_cache[crop]

    from src.data_loader import load_and_preprocess_data
    from src.scaling import inverse_scale_y, scale_with_fitted

    X_climate, X_soil, y = load_and_preprocess_data(
        target_crop=crop,
        soil_path=config.SOIL_CSV,
        climate_path=config.CLIMATE_CSV,
        yield_path=config.YIELD_CSV,
    )

    if len(y) == 0:
        raise PredictionError(f"No data found for crop '{crop}'")

    _, X_clim_val, _, X_soil_val, _, y_val = train_test_split(
        X_climate,
        X_soil,
        y,
        test_size=0.2,
        random_state=42,  # must match main.py / evaluate.py
    )

    X_clim_val, X_soil_val, _ = scale_with_fitted(
        X_clim_val,
        X_soil_val,
        y_val,
        save_dir=config.SAVE_DIR,
        target_crop=crop,
    )

    model = _get_model(crop)

    y_pred_scaled = model.predict(
        {"Climate_Input": X_clim_val, "Soil_Input": X_soil_val}, verbose=0
    ).flatten()
    y_pred = inverse_scale_y(y_pred_scaled, save_dir=config.SAVE_DIR, target_crop=crop)

    r2 = float(r2_score(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))

    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))

    stats = {
        "crop": crop,
        "crop_name": config.CROPS[crop],
        "r2": round(r2, 4),
        "rmse_mt_per_ha": round(rmse, 4),
        "mae_mt_per_ha": round(mae, 4),
        "mean_actual_mt_per_ha": round(float(np.mean(y_val)), 4),
        "n_samples": int(len(y)),
        "n_validation": int(len(y_val)),
        "model_parameters": int(model.count_params()),
        "architecture": buf.getvalue(),
    }

    with _lock:
        _stats_cache[crop] = stats
    return stats