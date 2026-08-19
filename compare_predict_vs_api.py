
"""
Cross-check the live FastAPI /predict endpoint against the same
prediction made directly through your training-time pipeline
(load_and_preprocess_data -> scale_with_fitted -> model -> inverse_scale_y).
 
If both paths agree (within tiny floating-point rounding), the backend
is faithfully reproducing your training pipeline.
 
Run from your PROJECT ROOT, with your normal venv (the one that has
tensorflow installed) active:
 
    python compare_predict_vs_api.py
 
Make sure the backend is running first:
    uvicorn backend.app.main:app --reload --port 8000
"""
 
import numpy as np
import requests
import tensorflow as tf
 
from src.data_loader import load_and_preprocess_data
from src.scaling import scale_with_fitted, inverse_scale_y
 
API_URL = "http://localhost:8000"
 
TARGET_CROP = "PD_Y"       # change to test other crops
TEST_DISTRICT = "arghakhanchi"  # same district Gemini's script tested
TEST_YEARS = [1981, 1982, 1983, 1984, 1985]  # same years Gemini's script tested
SAVE_DIR = "saved_models"
 
 
def main():
    print(f"Loading data + rebuilding samples for {TARGET_CROP} ...")
 
    # This mirrors preprocessing.py's loop closely enough to recover which
    # (district, year) each sample index corresponds to.
    import pandas as pd
    soil_df = pd.read_csv("data/processed/nepal_75_districts_soil_dataset.csv")
    climate_df = pd.read_csv("data/processed/nepal_district_climate_1979_2024.csv")
    yield_df = pd.read_csv("data/processed/crops-yield-1979-2024.csv")
 
    soil_df.columns = soil_df.columns.str.strip()
    climate_df.columns = climate_df.columns.str.strip()
    yield_df.columns = yield_df.columns.str.strip()
 
    climate_df["DATE"] = pd.to_datetime(climate_df["DATE"])
    climate_df["Year"] = climate_df["DATE"].dt.year.astype(int)
    yield_df["Year"] = yield_df["Date"].astype(str).str.split("-").str[0].str.strip().astype(int)
    yield_df = yield_df.sort_values(["District", "Year"]).reset_index(drop=True)
    yield_df["District"] = yield_df["District"].astype(str).str.strip().str.lower()
 
    # Recreate the *exact* filtering preprocessing.py applies, including the
    # "district not in soil data" skip — missing this was the bug in the
    # previous version of this script (it caused index drift).
    soil_districts = set(soil_df["District"].astype(str).str.strip().str.lower())
 
    # Reproduce the same district_to_coord matching preprocessing.py does,
    # so the "district not in district_to_coord" skip matches too.
    climate_coords = climate_df[["LAT", "LON"]].drop_duplicates().values
    district_to_coord = {}
    for _, row in soil_df.iterrows():
        dist = str(row["District"]).strip().lower()
        s_lat, s_lon = float(row["lat"]), float(row["lon"])
        distances = np.sqrt(
            (climate_coords[:, 0] - s_lat) ** 2 + (climate_coords[:, 1] - s_lon) ** 2
        )
        closest_idx = np.argmin(distances)
        district_to_coord[dist] = (
            climate_coords[closest_idx][0],
            climate_coords[closest_idx][1],
        )
 
    valid_rows = []
    for _, row in yield_df.iterrows():
        district = row["District"]
        target_yield = row[TARGET_CROP]
 
        if district not in soil_districts:
            continue
        if pd.isna(target_yield) or target_yield <= 0.0:
            continue
        if district not in district_to_coord:
            continue
 
        # Final check preprocessing.py applies: must have at least one
        # matching climate row for this district's grid point + year.
        c_lat, c_lon = district_to_coord[district]
        year = int(row["Year"])
        has_climate = (
            (
                np.isclose(climate_df["LAT"], c_lat)
                & np.isclose(climate_df["LON"], c_lon)
                & (climate_df["Year"] == year)
            )
            .any()
        )
        if not has_climate:
            continue
 
        valid_rows.append((district, year))
 
    print(f"Recovered {len(valid_rows)} valid (district, year) rows for {TARGET_CROP} "
          f"(should be well under {len(soil_districts)} districts x ~44 years).")
 
    def find_index(district, year):
        for i, (d, y) in enumerate(valid_rows):
            if d == district and y == year:
                return i
        return None
 
    # Load the actual model arrays (same call predict.py makes)
    X_climate, X_soil, y = load_and_preprocess_data(target_crop=TARGET_CROP)
 
    model = tf.keras.models.load_model(f"{SAVE_DIR}/best_{TARGET_CROP}_model.keras")
 
    print(f"\n{'District':<15}{'Year':<8}{'Local (mt/ha)':<16}{'API (mt/ha)':<16}{'Diff'}")
    print("-" * 65)
 
    for year in TEST_YEARS:
        idx = find_index(TEST_DISTRICT, year)
        if idx is None:
            print(f"{TEST_DISTRICT:<15}{year:<8} SKIPPED: no yield row for this district/year")
            continue
        district = TEST_DISTRICT
 
        # --- Local prediction: same path as predict.py ---
        clim, soil, _ = scale_with_fitted(
            X_climate[idx : idx + 1], X_soil[idx : idx + 1], None,
            save_dir=SAVE_DIR, target_crop=TARGET_CROP,
        )
        local_pred_scaled = model.predict(
            {"Climate_Input": clim, "Soil_Input": soil}, verbose=0
        ).flatten()
        local_pred = inverse_scale_y(local_pred_scaled, save_dir=SAVE_DIR, target_crop=TARGET_CROP)[0]
 
        # --- API prediction: same district/year, through the live server ---
        resp = requests.post(
            f"{API_URL}/predict",
            json={"crop": TARGET_CROP, "district": district, "year": year},
        )
        if resp.status_code != 200:
            print(f"{district:<15}{year:<8} API ERROR: {resp.text}")
            continue
 
        api_pred = resp.json()["predicted_yield_mt_per_ha"]
        diff = abs(local_pred - api_pred)
 
        print(f"{district:<15}{year:<8}{local_pred:<16.4f}{api_pred:<16.4f}{diff:.4f}")
 
    print("\nIf 'Diff' is ~0 (tiny floating point noise, e.g. < 0.001) for every")
    print("row, your live API is reproducing the training pipeline exactly.")
 
 
if __name__ == "__main__":
    main()
 
