# FRONTEND 

## React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

### React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

### Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


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
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

uvicorn backend.app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## 3. Endpoints

| Method | Path                  | Purpose                                              |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/health`              | Liveness check                                       |
| GET    | `/crops`                | List supported crops (PD_Y, MZ_Y, WH_Y) + display names |
| GET    | `/districts`            | List all districts with soil data (lowercased)        |
| GET    | `/years/{district}`     | Years with climate data available for that district   |
| GET    | `/stats`                | Per-crop model statistics (R², RMSE, MAE, architecture) |
| POST   | `/predict`               | Run a prediction                                     |

Saved model plots are also served at `/static/models/training_history.png`
and `/static/models/actual_vs_predicted.png`.

### `GET /stats`

Computes R², RMSE, MAE and the model architecture summary for each crop on
the same 20% validation split used by `evaluate.py`. Results are cached after
the first call (the first call can take a few seconds while the validation
predictions run).

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
`year` is optional — omit it or send `null` to use the most recent year with
climate data for that district. `land_area` (hectares) is optional — omit it or
send `null` if you only want yield per hectare.

Response:
```json
{
  "crop": "PD_Y",
  "crop_name": "Paddy (Rice)",
  "district": "kathmandu",
  "year": 2024,
  "predicted_yield_mt_per_ha": 3.42,
  "confidence_pct": 85.4,
  "error_margin_mt_per_ha": 0.37,
  "land_area_ha": 2.5,
  "predicted_total_yield_mt": 8.55
}
```
`confidence_pct` is derived from the crop's validation error relative to its
mean yield, and `error_margin_mt_per_ha` is the validation MAE.

Errors:
- `400` — unsupported crop code
- `404` — unknown district, or no climate data for that district/year
- `500` — missing model/scaler files, or an unexpected failure

## 4. Frontend wiring

The frontend has two main pages:

- **Prediction** (`/prediction`) — select a crop, a district, and enter a land
  area in hectares. It POSTs to `/predict` and shows the predicted yield
  (mt/ha), total production (mt), and the model confidence.
- **Statistics** (`/statistics`) — fetches `/stats` and renders R², RMSE, MAE,
  sample counts and the architecture of each crop's model, plus the saved
  training plots.

`frontend/src/lib/api.js` wraps all backend endpoints (`getCrops`,
`getDistricts`, `getStats`, `predictYield`) and exposes the plot URLs.

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
