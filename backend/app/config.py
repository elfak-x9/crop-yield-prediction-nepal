"""
Configuration for the crop-yield-prediction backend.

All values here are copied directly from your existing pipeline
(src/preprocessing.py, src/scaling.py, main.py) so predictions match
what the models were trained on.
"""

import os

# --- Paths ---------------------------------------------------------------
# backend/ is expected to sit at the ROOT of your project, as a sibling of
# src/, saved_models/, data/, main.py, predict.py, frontend/.
#
#   crop-yield-prediction-nepal/
#     |-- backend/          <- this app
#     |-- src/
#     |-- saved_models/
#     |-- data/processed/
#     |-- frontend/
#     `-- main.py
#
# Override with env vars if your layout differs.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(BACKEND_DIR))

SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(PROJECT_ROOT, "saved_models"))

SOIL_CSV = os.environ.get(
    "SOIL_CSV",
    os.path.join(PROJECT_ROOT, "data", "processed", "nepal_75_districts_soil_dataset.csv"),
)
CLIMATE_CSV = os.environ.get(
    "CLIMATE_CSV",
    os.path.join(PROJECT_ROOT, "data", "processed", "nepal_district_climate_1979_2024.csv"),
)

# --- Model / sequence config (must match main.py / preprocessing.py) -----
SEQUENCE_LENGTH = 150

# Exact column order used when building X_soil in src/preprocessing.py
SOIL_COLS = [
    "ph",
    "organic_matter_pct",
    "total_nitrogen_pct",
    "p2o5_kgha",
    "potassium_kgha",
    "zinc_ppm",
    "boron_ppm",
    "sand_pct",
    "clay_pct",
    "silt_pct",
]

# Exact column order used when building X_climate in src/preprocessing.py
CLIMATE_FEATURES = ["PRECTOT", "RH2M", "T2M", "T2M_MAX", "T2M_MIN", "WS10M"]

# --- Supported crops -------------------------------------------------------
# Codes must match TARGET_CROP values used when training (main.py) and the
# best_{CODE}_model.keras / {CODE}_scaler_*.pkl filenames in saved_models/.
CROPS = {
    "PD_Y": "Paddy (Rice)",
    "MZ_Y": "Maize",
    "WH_Y": "Wheat",
    "BW_Y": "Buckwheat",
}

# --- CORS ------------------------------------------------------------------
FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
