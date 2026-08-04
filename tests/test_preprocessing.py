# tests/test_preprocessing.py — Unit tests for data preprocessing & leakage prevention
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALLOWED_FEATURES,
    BANNED_COLUMNS,
    TARGET_COLUMNS,
    TEMPORAL_SPLIT_DATE,
)
from src.preprocessing import (
    calculate_trip_duration,
    drop_banned_columns,
    filter_outliers,
    perform_temporal_split,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing a mock raw DataFrame with valid and outlier rows."""
    data = {
        "VendorID": [1, 2, 1, 2, 1],
        "tpep_pickup_datetime": [
            "2022-05-10 10:00:00",  # Valid train
            "2022-05-20 14:30:00",  # Valid train
            "2022-05-25 18:00:00",  # Valid test
            "2022-04-30 23:59:00",  # Outlier: out of May 2022
            "2022-05-15 12:00:00",  # Outlier: negative fare
        ],
        "tpep_dropoff_datetime": [
            "2022-05-10 10:15:00",  # 15 min duration
            "2022-05-20 15:00:00",  # 30 min duration
            "2022-05-25 18:45:00",  # 45 min duration
            "2022-05-01 00:10:00",  # 11 min duration
            "2022-05-15 12:10:00",  # 10 min duration
        ],
        "passenger_count": [1.0, 2.0, 1.0, 1.0, 1.0],
        "trip_distance": [2.5, 5.0, 8.0, 1.0, 3.0],
        "RatecodeID": [1.0, 1.0, 2.0, 1.0, 1.0],
        "store_and_fwd_flag": ["N", "N", "N", "N", "N"],
        "PULocationID": [132, 230, 142, 100, 100],
        "DOLocationID": [236, 161, 132, 100, 100],
        "payment_type": [1, 1, 2, 1, 1],
        "fare_amount": [12.50, 25.00, 52.00, 10.00, -5.00],  # -5.00 is outlier
        "extra": [0.5, 0.5, 0.0, 0.5, 0.5],
        "mta_tax": [0.5, 0.5, 0.5, 0.5, 0.5],
        "tip_amount": [2.0, 4.0, 10.0, 0.0, 0.0],
        "tolls_amount": [0.0, 0.0, 6.55, 0.0, 0.0],
        "improvement_surcharge": [0.3, 0.3, 0.3, 0.3, 0.3],
        "total_amount": [15.8, 30.3, 69.35, 11.3, -3.7],
        "congestion_surcharge": [2.5, 2.5, 2.5, 2.5, 2.5],
        "airport_fee": [0.0, 0.0, 1.25, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    return df


def test_calculate_trip_duration(sample_raw_data: pd.DataFrame):
    """Test that trip duration in minutes is accurately computed."""
    df_dur = calculate_trip_duration(sample_raw_data)
    assert "duration_minutes" in df_dur.columns
    assert df_dur.loc[0, "duration_minutes"] == pytest.approx(15.0)
    assert df_dur.loc[1, "duration_minutes"] == pytest.approx(30.0)
    assert df_dur.loc[2, "duration_minutes"] == pytest.approx(45.0)


def test_filter_outliers(sample_raw_data: pd.DataFrame):
    """Test outlier filtering rules for timestamp and negative fare."""
    df_dur = calculate_trip_duration(sample_raw_data)
    filtered_df, stats = filter_outliers(df_dur, verbose=False)

    # Rows index 3 (April timestamp) and index 4 (negative fare) should be dropped
    assert len(filtered_df) == 3
    assert stats["out_of_bound_timestamps"] == 1
    assert stats["invalid_fares"] == 1
    assert stats["total_dropped_rows"] == 2


def test_drop_banned_columns_leakage_prevention(sample_raw_data: pd.DataFrame):
    """CRITICAL LEAKAGE TEST: Assert no banned column reaches the feature frame."""
    df_dur = calculate_trip_duration(sample_raw_data)
    feature_df = drop_banned_columns(df_dur)

    # 1. Assert NONE of the banned columns exist
    for banned_col in BANNED_COLUMNS:
        assert (
            banned_col not in feature_df.columns
        ), f"Leakage violation: {banned_col} found!"

    # 2. Assert ONLY allowed features + target columns exist
    expected_set = set(ALLOWED_FEATURES + TARGET_COLUMNS)
    actual_set = set(feature_df.columns)
    assert actual_set == expected_set


def test_perform_temporal_split(sample_raw_data: pd.DataFrame):
    """Test that temporal split cleanly divides train (< split_date) and test (>= split_date)."""
    df_dur = calculate_trip_duration(sample_raw_data)
    filtered_df, _ = filter_outliers(df_dur, verbose=False)
    feature_df = drop_banned_columns(filtered_df)

    train_df, test_df = perform_temporal_split(feature_df, TEMPORAL_SPLIT_DATE)
    split_dt = pd.to_datetime(TEMPORAL_SPLIT_DATE)

    assert len(train_df) == 2  # May 10 and May 20
    assert len(test_df) == 1  # May 25

    assert (pd.to_datetime(train_df["tpep_pickup_datetime"]) < split_dt).all()
    assert (pd.to_datetime(test_df["tpep_pickup_datetime"]) >= split_dt).all()
