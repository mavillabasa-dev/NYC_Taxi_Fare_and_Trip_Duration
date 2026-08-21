"""Input and output schemas for the model."""

from datetime import datetime

from pydantic import BaseModel, Field

# Validation Constraints & Boundaries
MIN_LOCATION_ID = 1
MAX_LOCATION_ID = 265
MIN_PASSENGER_COUNT = 1
MAX_PASSENGER_COUNT = 9
MIN_RATECODE_ID = 1
MAX_RATECODE_ID = 6
MIN_TRIP_DISTANCE = 0.0
MAX_TRIP_DISTANCE = 150.0


class PredictionRequest(BaseModel):
	PULocationID: int = Field(ge=MIN_LOCATION_ID, le=MAX_LOCATION_ID)
	DOLocationID: int = Field(ge=MIN_LOCATION_ID, le=MAX_LOCATION_ID)
	tpep_pickup_datetime: datetime
	passenger_count: int = Field(ge=MIN_PASSENGER_COUNT, le=MAX_PASSENGER_COUNT)
	RatecodeID: int = Field(ge=MIN_RATECODE_ID, le=MAX_RATECODE_ID)
	trip_distance: float = Field(gt=MIN_TRIP_DISTANCE, le=MAX_TRIP_DISTANCE)


class PredictionResponse(BaseModel):
	predicted_fare: float
	predicted_duration_minutes: float
	model_version: str


class HealthResponse(BaseModel):
	status: str
	model_loaded: bool
	model_version: str | None = None
	detail: str | None = None
