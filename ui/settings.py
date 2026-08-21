"""Environment configuration and constants for the dashboard."""

import os
from dataclasses import dataclass

# Form & Input Defaults
DEFAULT_PICKUP_ZONE_ID = 132  # JFK Airport (Queens)
DEFAULT_DROPOFF_ZONE_ID = 236  # Upper East Side North (Manhattan)
DEFAULT_PICKUP_DATETIME = "2022-05-20T14:30:00"
DEFAULT_PASSENGER_COUNT = 1
DEFAULT_RATECODE = 1
DEFAULT_TRIP_DISTANCE_MILES = 3.0

# Form Validation Constraints & Bounds
PASSENGER_MIN = 1
PASSENGER_MAX = 9
RATECODES = [1, 2, 3, 4, 5, 6]
TRIP_DISTANCE_MIN = 0.1
TRIP_DISTANCE_MAX = 150.0

# Map & Visualization Constants
DEFAULT_MAP_CENTER = {"lat": 40.75, "lon": -73.95}
DEFAULT_CHOROPLETH_ZOOM = 9
DEFAULT_POINT_MAP_ZOOM = 10
DEFAULT_MAP_OPACITY = 0.7
CHOROPLETH_HEIGHT = 500
POINT_MAP_HEIGHT = 400
PICKUP_MARKER_COLOR = "#2ecc71"
DROPOFF_MARKER_COLOR = "#e74c3c"
MARKER_SIZE = 14

# HTTP Status Codes
HTTP_STATUS_UNREACHABLE = 0
HTTP_STATUS_OK = 200
HTTP_STATUS_UNPROCESSABLE = 422
HTTP_STATUS_UNAVAILABLE = 503


@dataclass(frozen=True)
class Settings:
	api_url: str = os.getenv("API_URL", "http://localhost:8000")
	dataset_dir: str = os.getenv("DATASET_DIR", "../dataset")


settings = Settings()
