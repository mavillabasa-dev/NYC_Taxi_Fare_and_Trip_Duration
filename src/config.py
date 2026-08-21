# src/config.py — Central configuration, paths, constants, and random seed
import os

# Project Root Directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories & File Paths
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
RAW_DATA_PATH = os.path.join(DATASET_DIR, "yellow_tripdata_2022-05.parquet")
TRAIN_CLEANED_PATH = os.path.join(DATASET_DIR, "train_cleaned.parquet")
TEST_CLEANED_PATH = os.path.join(DATASET_DIR, "test_cleaned.parquet")

# Ingestion Source URLs & Local Auxiliary Paths
YELLOW_TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-05.parquet"
)
TAXI_ZONE_LOOKUP_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
)
TAXI_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

TAXI_ZONE_LOOKUP_PATH = os.path.join(DATASET_DIR, "taxi_zone_lookup.csv")
TAXI_ZONE_SHAPEFILE_PATH = os.path.join(DATASET_DIR, "taxi_zones.zip")
TAXI_ZONE_CENTROIDS_PATH = os.path.join(DATASET_DIR, "taxi_zone_centroids.csv")

# Data Download Guardrails & Validation Thresholds
MIN_PARQUET_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB
MIN_PARQUET_ROW_COUNT = 3_000_000
MIN_LOOKUP_SIZE_BYTES = 10 * 1024  # 10 KB
MIN_LOOKUP_ROW_COUNT = 260
MIN_SHAPEFILE_SIZE_BYTES = 500 * 1024  # 500 KB

# Models Directory & Serialized Paths
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

# Random Seed Constant (Enforced across all scripts)
RANDOM_SEED = 42

# Temporal Train/Test Split Boundary Constant (First ~3 weeks train, last week test)
TEMPORAL_SPLIT_DATE = "2022-05-23 00:00:00"

# Feature Contract: Allowed Prediction-Time Features
ALLOWED_FEATURES = [
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
    "VendorID",
]

# Target Columns
TARGET_COLUMNS = [
    "fare_amount",
    "duration_minutes",
]

# Feature Contract: Banned Post-Trip Leakage Columns
BANNED_COLUMNS = [
    "tpep_dropoff_datetime",
    "total_amount",
    "tip_amount",
    "tolls_amount",
    "extra",
    "mta_tax",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "payment_type",
    "store_and_fwd_flag",
]

# Data Cleaning Thresholds (T-103 EDA Baseline)
MAY_2022_START = "2022-05-01 00:00:00"
MAY_2022_END = "2022-06-01 00:00:00"
MIN_FARE_AMOUNT = 2.50
MAX_FARE_AMOUNT = 500.00
MIN_DURATION_MINUTES = 0.5
MAX_DURATION_MINUTES = 360.0  # 6 hours
MIN_TRIP_DISTANCE = 0.01  # > 0 miles
MAX_TRIP_DISTANCE = 150.0  # max reasonable NYC trip
MIN_PASSENGER_COUNT = 1
MAX_PASSENGER_COUNT = 9
VALID_RATECODES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
MIN_LOCATION_ID = 1
MAX_LOCATION_ID = 265

# Geodetic & Spatial Feature Constants
EARTH_RADIUS_MILES = 3958.8
MILES_PER_DEGREE_LATITUDE = 69.0
MAX_HAVERSINE_RATIO = 10.0
EPSILON_DISTANCE = 0.001

# Airport Zone & Ratecode IDs
JFK_AIRPORT_ZONE_ID = 132
NEWARK_AIRPORT_ZONE_ID = 1
JFK_RATECODE_ID = 2
NEWARK_RATECODE_ID = 3

# Temporal & Rush Hour Constants
AM_RUSH_START_HOUR = 7
AM_RUSH_END_HOUR = 9
PM_RUSH_START_HOUR = 16
PM_RUSH_END_HOUR = 19
MEMORIAL_DAY_MONTH = 5
MEMORIAL_DAY_DAY = 30
HOURS_IN_DAY = 24.0
DAYS_IN_WEEK = 7.0

# Target Encoding Defaults
DEFAULT_TARGET_ENCODING_SMOOTHING = 10.0
DEFAULT_GLOBAL_FARE_MEAN = 15.15
DEFAULT_VENDOR_ID = 2
