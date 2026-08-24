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
DEFAULT_MAP_OPACITY = 0.78
CHOROPLETH_HEIGHT = 500
POINT_MAP_HEIGHT = 400

# Carto's dark base map. Free like positron, no Mapbox token, but it lets the gold
# marks carry the image instead of competing with a bright basemap.
MAP_STYLE = "carto-darkmatter"

# Pickup is the brighter of the two: it is where the user is. The text labels on the
# markers carry the identity, so colour only has to separate them, not encode them.
PICKUP_MARKER_COLOR = "#FFC72C"  # taxi yellow
DROPOFF_MARKER_COLOR = "#F2ECE0"  # cream
TRIP_LINE_COLOR = "#F0BE4A"  # gold
MARKER_SIZE = 15

# One hue ramp, dark to bright, matching the deck's treatment of single-series charts.
# Colour never re-encodes anything position or length already shows, so a categorical
# palette would be wrong here: fare is one continuous quantity.
FARE_COLOR_SCALE = ["#3D3020", "#6B5220", "#9A7526", "#C89A33", "#F0BE4A", "#FFC72C"]

# Chart furniture, from ppt/index.html.
MAP_TEXT_COLOR = "#F2ECE0"
MAP_MUTED_COLOR = "#A9A294"
MAP_FONT = "JetBrains Mono, Consolas, monospace"

# HTTP Status Codes
HTTP_STATUS_UNREACHABLE = 0
HTTP_STATUS_OK = 200
HTTP_STATUS_UNPROCESSABLE = 422
HTTP_STATUS_UNAVAILABLE = 503


def _get_default_dataset_dir() -> str:
    if os.path.exists("dataset"):
        return "dataset"
    repo_root_dataset = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
    )
    if os.path.exists(repo_root_dataset):
        return repo_root_dataset
    return "../dataset"


@dataclass(frozen=True)
class Settings:
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    dataset_dir: str = os.getenv("DATASET_DIR", _get_default_dataset_dir())


settings = Settings()
