"""Input and output schemas for the model."""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
	PULocationID: int = Field(ge=1, le=265)
	DOLocationID: int = Field(ge=1, le=265)
	tpep_pickup_datetime: datetime
	passenger_count: int = Field(ge=1, le=9)
	RatecodeID: int = Field(ge=1, le=6)
	trip_distance: float = Field(gt=0, le=150)


class PredictionResponse(BaseModel):
	predicted_fare: float
	predicted_duration_minutes: float
	model_version: str


class HealthResponse(BaseModel):
	status: str
	model_loaded: bool
	model_version: str | None = None
	detail: str | None = None
