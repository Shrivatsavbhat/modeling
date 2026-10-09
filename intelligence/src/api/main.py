"""Standalone forecasting service; no Supabase or main-backend dependency."""
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
from ml.predict import DEFAULT_MODEL, load_artifact, predict_demand, predict_from_real_history

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hospital_id: str = Field(min_length=1)
    medicine_id: str = Field(min_length=1)
    horizon_days: Literal[7, 14] = 7
    outbreak_flag: int = Field(default=0, ge=0, le=1)


class HistoryPoint(BaseModel):
    date: date
    quantity_requested: float = Field(ge=0)
    outbreak_flag: int = Field(default=0, ge=0, le=1)


class RealHistoryPredictionRequest(BaseModel):
    hospital_id: str = Field(min_length=1)
    medicine_id: str = Field(min_length=1)
    horizon_days: Literal[7, 14] = 7
    outbreak_flag: int = Field(default=0, ge=0, le=1)
    history: list[HistoryPoint] = Field(min_length=28)


class DailyPrediction(BaseModel):
    date: date
    predicted_demand: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    hospital_id: str
    medicine_id: str
    model_version: str
    horizon_days: Literal[7, 14]
    outbreak_flag: int
    baseline_daily_demand: float = Field(ge=0, allow_inf_nan=False)
    average_predicted_demand: float = Field(ge=0, allow_inf_nan=False)
    surge_percent: float
    risk_level: Literal["normal", "elevated", "high_surge", "critical_surge"]
    predictions: list[DailyPrediction]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_version: str
    training_date_range: dict[str, str]
    supported_hospitals: list[str]
    supported_medicines: list[str]
    supported_combinations: list[dict[str, str]]
    feature_names: list[str]
    dataset_origin: str
    evaluation_metrics: dict


def create_app(model_path: str | Path = DEFAULT_MODEL) -> FastAPI:
    """Cache one artifact per service instance; restart after replacing the model."""
    service = FastAPI(title="PulseGrid AI — Demand Forecasting V1", version="1.0.0")

    @lru_cache(maxsize=1)
    def get_model() -> dict:
        try:
            return load_artifact(model_path)
        except (FileNotFoundError, ValueError, KeyError, EOFError) as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    @service.get("/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "ok"}

    @service.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> dict:
        saved = get_model()
        return dict(model_version=saved["model_version"], training_date_range=saved["training_date_range"],
            supported_hospitals=saved["categories"]["hospital_id"],
            supported_medicines=saved["categories"]["medicine_id"],
            supported_combinations=saved["history"][["hospital_id", "medicine_id"]].drop_duplicates().to_dict("records"),
            feature_names=saved["feature_names"], dataset_origin=saved["evaluation"]["dataset_origin"],
            evaluation_metrics=saved["evaluation"])

    @service.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> dict:
        saved = get_model()
        try:
            return predict_demand(request.hospital_id, request.medicine_id,
                                  request.horizon_days, request.outbreak_flag, artifact=saved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @service.post("/predict/real-history")
    def predict_real_history(request: RealHistoryPredictionRequest) -> dict:
        frame = pd.DataFrame([
            {
                "date": point.date,
                "hospital_id": request.hospital_id,
                "medicine_id": request.medicine_id,
                "quantity_requested": point.quantity_requested,
                "outbreak_flag": point.outbreak_flag,
            }
            for point in request.history
        ])
        try:
            return predict_from_real_history(frame, request.horizon_days, request.outbreak_flag)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return service


app = create_app()
