# src/config.py — Central configuration, paths, constants, and random seed
import os

# Project Root Directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories & File Paths
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
RAW_DATA_PATH = os.path.join(DATASET_DIR, "yellow_tripdata_2022-05.parquet")
TRAIN_CLEANED_PATH = os.path.join(DATASET_DIR, "train_cleaned.parquet")
TEST_CLEANED_PATH = os.path.join(DATASET_DIR, "test_cleaned.parquet")

# Models Directory
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
