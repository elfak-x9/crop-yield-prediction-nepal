from typing import Optional

from pydantic import BaseModel, Field


class CropInfo(BaseModel):
    code: str
    name: str


class PredictRequest(BaseModel):
    crop: str = Field(..., description="Crop code, e.g. 'PD_Y', 'MZ_Y', 'WH_Y', 'BW_Y'")
    district: str = Field(..., description="District name, e.g. 'Kathmandu'")
    year: int = Field(..., description="Year to predict for, must exist in the climate data")
    land_area: Optional[float] = Field(
        None, gt=0, description="Optional land area in hectares, to compute total yield"
    )


class PredictResponse(BaseModel):
    crop: str
    crop_name: str
    district: str
    year: int
    predicted_yield_mt_per_ha: float
    land_area_ha: Optional[float] = None
    predicted_total_yield_mt: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
