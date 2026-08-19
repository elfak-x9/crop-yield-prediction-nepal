from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config, inference
from .schemas import CropInfo, PredictRequest, PredictResponse

app = FastAPI(
    title="Nepal Crop Yield Prediction API",
    description="Serves the trained dual-input CNN-LSTM crop yield models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/crops", response_model=list[CropInfo])
def get_crops():
    return [CropInfo(code=code, name=name) for code, name in config.CROPS.items()]


@app.get("/districts", response_model=list[str])
def get_districts():
    try:
        return inference.list_districts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Soil data file not found: {exc}") from exc


@app.get("/years/{district}", response_model=list[int])
def get_years(district: str):
    try:
        return inference.list_years_for_district(district)
    except inference.PredictionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.crop not in config.CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported crop '{req.crop}'. Supported: {list(config.CROPS.keys())}",
        )

    try:
        yield_mt_ha = inference.predict_yield(req.crop, req.district, req.year)
    except inference.PredictionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # model/scaler load failures, unexpected shape issues, etc.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    total = yield_mt_ha * req.land_area if req.land_area else None

    return PredictResponse(
        crop=req.crop,
        crop_name=config.CROPS[req.crop],
        district=req.district.strip().lower(),
        year=req.year,
        predicted_yield_mt_per_ha=round(yield_mt_ha, 4),
        land_area_ha=req.land_area,
        predicted_total_yield_mt=round(total, 4) if total is not None else None,
    )
