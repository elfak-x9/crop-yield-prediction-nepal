# Crop Yield Prediction — Backend

A FastAPI service that wraps your trained dual-input CNN-LSTM models
(`saved_models/best_{CROP}_model.keras`) and serves live predictions to
the React/Vite frontend.

It reuses your existing feature logic exactly:
- Soil columns and one-hot `parent_soil` encoding from `src/preprocessing.py`
- Climate columns (`PRECTOT, RH2M, T2M, T2M_MAX, T2M_MIN, WS10M`) and the
  150-step sequence padding/truncation logic from `src/preprocessing.py`
- Per-crop `StandardScaler`s (`{CROP}_scaler_climate.pkl`,
  `{CROP}_scaler_soil.pkl`, `{CROP}_scaler_y.pkl`) from `src/scaling.py`

No training code was touched — this only does inference.

## 1. Where this folder goes

Drop `backend/` into the root of your project, next to `src/`,
`saved_models/`, `data/`, `main.py`, and `frontend/`:

```
crop-yield-prediction-nepal/
├── backend/              <- this folder
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── inference.py
│   │   ├── main.py
│   │   └── schemas.py
│   └── requirements.txt
├── src/
├── saved_models/
├── data/processed/
├── frontend/
└── main.py
```

The backend finds `saved_models/` and `data/processed/*.csv` automatically
based on this layout (`backend/app/config.py` walks up two directories).
If your layout differs, set env vars instead:

```
SAVE_DIR=/path/to/saved_models
SOIL_CSV=/path/to/nepal_75_districts_soil_dataset.csv
CLIMATE_CSV=/path/to/nepal_district_climate_1979_2024.csv
```

## 2. Install & run

From the project root (important — so the relative default paths resolve):

```bash
cd crop-yield-prediction-nepal
python -m venv backend/.venv
# Windows:
backend\.venv\Scripts\activate
# macOS/Linux:
source backend/.venv/bin/activate

pip install -r backend/requirements.txt

uvicorn backend.app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## 3. Endpoints

| Method | Path                  | Purpose                                              |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/health`              | Liveness check                                       |
| GET    | `/crops`                | List supported crop codes + display names            |
| GET    | `/districts`            | List all 75 districts (lowercased, as used internally)|
| GET    | `/years/{district}`     | Years with climate data available for that district   |
| POST   | `/predict`               | Run a prediction                                     |

### `POST /predict`

Request body:
```json
{
  "crop": "PD_Y",
  "district": "kathmandu",
  "year": 2020,
  "land_area": 2.5
}
```
`land_area` (hectares) is optional — omit it or send `null` if you only
want yield per hectare.

Response:
```json
{
  "crop": "PD_Y",
  "crop_name": "Paddy (Rice)",
  "district": "kathmandu",
  "year": 2020,
  "predicted_yield_mt_per_ha": 3.42,
  "land_area_ha": 2.5,
  "predicted_total_yield_mt": 8.55
}
```

Errors:
- `400` — unsupported crop code
- `404` — unknown district, or no climate data for that district/year
- `500` — missing model/scaler files, or an unexpected failure

## 4. Frontend wiring

`frontend/src/lib/api.js` was added, and
`frontend/src/components/prediction/predictionform.jsx` was updated to:
- load the crop list and district list from the backend on mount,
- load the available years once a district is picked,
- POST to `/predict` and show the real result instead of the old
  hardcoded "5.8 Tons (Demo)" value.

Copy `frontend/.env.example` to `frontend/.env` and adjust
`VITE_API_URL` if your backend isn't on `localhost:8000`.

Run both dev servers side by side:
```bash
# terminal 1
uvicorn backend.app.main:app --reload --port 8000

# terminal 2
cd frontend
npm install
npm run dev
```

## 5. Notes / things to double check

- **Model loading is lazy and cached per crop** — the first prediction
  for a given crop will be slower (loading the `.keras` file + 3
  scalers); subsequent calls reuse the cached model.
- **Years available depend on your climate CSV** (`1979_2024` per the
  filename) — there's no future-year forecasting here, only prediction
  for years that exist in the historical climate data.
- If you add a 5th crop later, just add its code + display name to
  `CROPS` in `backend/app/config.py` — nothing else needs to change, as
  long as `best_{CODE}_model.keras` and `{CODE}_scaler_*.pkl` exist in
  `saved_models/`.
