from typing import Optional

from pydantic import BaseModel, Field


class CropInfo(BaseModel):
    code: str
    name: str


class PredictRequest(BaseModel):
    crop: str = Field(..., description="Crop code, e.g. 'PD_Y', 'MZ_Y', 'WH_Y'")
    district: str = Field(..., description="District name, e.g. 'Kathmandu'")
    year: Optional[int] = Field(
        None, description="Year to predict for; defaults to the most recent year available for the district"
    )
    land_area: Optional[float] = Field(
        None, gt=0, description="Land area in hectares, to compute total yield"
    )


class PredictResponse(BaseModel):
    crop: str
    crop_name: str
    district: str
    year: int
    predicted_yield_mt_per_ha: float
    confidence_pct: float
    error_margin_mt_per_ha: float
    is_projection: bool = False
    land_area_ha: Optional[float] = None
    predicted_total_yield_mt: Optional[float] = None


class CropStats(BaseModel):
    crop: str
    crop_name: str
    r2: float
    rmse_mt_per_ha: float
    mae_mt_per_ha: float
    mean_actual_mt_per_ha: float
    n_samples: int
    n_validation: int
    model_parameters: int
    architecture: str
    plot_url: str


class ErrorResponse(BaseModel):
    detail: str
