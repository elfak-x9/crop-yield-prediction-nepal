from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config, inference
from .schemas import CropInfo, CropStats, PredictRequest, PredictResponse

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

# Serve saved model plots (training_history.png, actual_vs_predicted.png)
app.mount("/static/models", StaticFiles(directory=config.SAVE_DIR), name="saved-model-plots")


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


@app.get("/stats", response_model=list[CropStats])
def get_stats():
    """Per-crop model statistics (R², RMSE, MAE) computed on the validation split."""
    results = []
    for crop in config.CROPS:
        try:
            results.append(inference.compute_crop_stats(crop))
        except inference.PredictionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return results


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Print formatted request data to the terminal
    print("\n==========================================")
    print("      NEW PREDICTION REQUEST RECEIVED     ")
    print("==========================================")
    print(f"Crop Code : {req.crop}")
    print(f"District  : {req.district}")
    print(f"Year      : {req.year if req.year is not None else '(default: latest)'}")
    print(f"Land Area : {req.land_area} ha" if req.land_area else "Land Area : Not specified")
    print("------------------------------------------")

    if req.crop not in config.CROPS:
        print(f"[ERROR] Unsupported crop '{req.crop}'")
        print("==========================================\n")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported crop '{req.crop}'. Supported: {list(config.CROPS.keys())}",
        )

    district = req.district.strip().lower()

    try:
        year = req.year if req.year is not None else inference.latest_year(district)
        pred_info = inference.predict_with_confidence(req.crop, district, year)
    except inference.PredictionError as exc:
        print(f"[ERROR] Prediction error: {exc}")
        print("==========================================\n")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # model/scaler load failures, unexpected shape issues, etc.
        print(f"[ERROR] Internal prediction failure: {exc}")
        print("==========================================\n")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    yield_mt_ha = pred_info["yield_mt_per_ha"]
    total = yield_mt_ha * req.land_area if req.land_area else None

    # Print calculated results to the terminal
    print(f"Crop Name          : {config.CROPS[req.crop]}")
    print(f"Year (used)        : {year}{' (projected climate)' if pred_info['is_projection'] else ''}")
    print(f"Yield (MT/ha)      : {round(yield_mt_ha, 4)}")
    print(f"Confidence         : {pred_info['confidence_pct']}% (+/- {pred_info['error_margin_mt_per_ha']})")
    if total is not None:
        print(f"Total Yield (MT)   : {round(total, 4)}")
    print("==========================================\n")

    return PredictResponse(
        crop=req.crop,
        crop_name=config.CROPS[req.crop],
        district=district,
        year=year,
        predicted_yield_mt_per_ha=round(yield_mt_ha, 4),
        confidence_pct=pred_info["confidence_pct"],
        error_margin_mt_per_ha=pred_info["error_margin_mt_per_ha"],
        is_projection=pred_info["is_projection"],
        land_area_ha=req.land_area,
        predicted_total_yield_mt=round(total, 4) if total is not None else None,
    )