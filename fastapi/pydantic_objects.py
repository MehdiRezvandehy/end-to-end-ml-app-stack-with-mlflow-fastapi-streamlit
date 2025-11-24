from typing import List
from pydantic import BaseModel, Field


class BuildingEnergyLoadPredictor(BaseModel):
    rel_compact: float = Field(..., ge=0.1, le=0.98, description="Relative Compactness")
    surface_area: float = Field(..., ge=100.0, le=1500.0, description="Surface Area")
    wall_area: float = Field(..., ge=50.0, le=800.0, description="Wall Area")
    roof_area: float = Field(..., ge=20, le=500.0, description="Roof Area")

    overall_height: float = Field(..., ge=3.5, le=5.0, description="Overall Height")
    orientation: int = Field(..., ge=2, le=5, description="Orientation")

    glazing_area: float = Field(..., ge=0, le=0.8, description="Glazing Area")
    glazing_dist: int = Field(..., ge=0, le=5,description="Glazing Area Distribution")


class PredictionResponse(BaseModel):
    predict_load_probability: float
