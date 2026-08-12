# tests/test_features.py — Unit & Integration tests for Feature Engineering Pipeline (T-105)
import os
import pickle
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALLOWED_FEATURES,
    BANNED_COLUMNS,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.features import (
    NYCFeaturePipeline,
    SpatialZoneFeatureExtractor,
    TargetCategoricalEncoder,
    TemporalFeatureExtractor,
    calculate_haversine_distance,
    calculate_manhattan_distance,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing a sample DataFrame conforming to ALLOWED_FEATURES."""
    return pd.DataFrame(
        [
            {
                "tpep_pickup_datetime": "2022-05-16 08:30:00",  # Monday 8:30 AM Rush Hour
                "PULocationID": 132,  # JFK Airport
                "DOLocationID": 236,  # Upper East Side
                "passenger_count": 1,
                "RatecodeID": 2,  # JFK Flat Rate
                "trip_distance": 15.5,
                "VendorID": 1,
            },
            {
                "tpep_pickup_datetime": "2022-05-21 22:15:00",  # Saturday 10:15 PM Weekend
                "PULocationID": 236,
                "DOLocationID": 236,  # Same Zone Trip
                "passenger_count": 2,
                "RatecodeID": 1,  # Standard Rate
                "trip_distance": 0.8,
                "VendorID": 2,
            },
            {
                "tpep_pickup_datetime": "2022-05-30 12:00:00",  # Memorial Day Holiday
                "PULocationID": 1,  # Newark Airport
                "DOLocationID": 132,  # JFK
                "passenger_count": 3,
                "RatecodeID": 3,  # Newark Rate
                "trip_distance": 22.0,
                "VendorID": 1,
            },
        ]
    )


def test_haversine_and_manhattan_calculations():
    """Test mathematical accuracy of Haversine and Manhattan distance functions."""
    # JFK (40.6413, -73.7781) to Times Square (40.7580, -73.9855)
    lat1, lon1 = 40.6413, -73.7781
    lat2, lon2 = 40.7580, -73.9855

    hav_dist = calculate_haversine_distance(lat1, lon1, lat2, lon2)
    man_dist = calculate_manhattan_distance(lat1, lon1, lat2, lon2)

    assert 13.0 <= hav_dist <= 14.0
    assert (
        man_dist >= hav_dist
    )  # Manhattan distance >= straight-line Haversine distance


def test_temporal_feature_extractor(sample_raw_dataframe):
    """Test temporal feature extraction, cyclical encodings, and rush hour/holiday flags."""
    extractor = TemporalFeatureExtractor()
    df_out = extractor.transform(sample_raw_dataframe)

    # Check extracted columns
    assert "pickup_hour" in df_out.columns
    assert "sin_hour" in df_out.columns
    assert "cos_hour" in df_out.columns
    assert "is_weekend" in df_out.columns
    assert "is_rush_hour" in df_out.columns
    assert "is_holiday" in df_out.columns

    # Row 0: Mon May 16 08:30 AM -> Hour 8, Rush hour = 1, Weekend = 0, Holiday = 0
    row0 = df_out.iloc[0]
    assert row0["pickup_hour"] == 8
    assert row0["is_weekend"] == 0
    assert row0["is_rush_hour"] == 1
    assert row0["is_holiday"] == 0

    # Row 1: Sat May 21 22:15 PM -> Hour 22, Rush hour = 0, Weekend = 1, Holiday = 0
    row1 = df_out.iloc[1]
    assert row1["pickup_hour"] == 22
    assert row1["is_weekend"] == 1
    assert row1["is_rush_hour"] == 0

    # Row 2: Mon May 30 (Memorial Day) -> Holiday = 1
    row2 = df_out.iloc[2]
    assert row2["is_holiday"] == 1


def test_spatial_zone_feature_extractor(sample_raw_dataframe):
    """Test spatial centroid lookup, distance metrics, same-zone and airport flags."""
    extractor = SpatialZoneFeatureExtractor()
    extractor.fit(sample_raw_dataframe)
    df_out = extractor.transform(sample_raw_dataframe)

    assert "haversine_distance" in df_out.columns
    assert "manhattan_distance" in df_out.columns
    assert "haversine_ratio" in df_out.columns
    assert "is_same_zone" in df_out.columns
    assert "is_jfk" in df_out.columns
    assert "is_newark" in df_out.columns

    # Row 0: JFK RatecodeID 2 -> is_jfk == 1
    assert df_out.iloc[0]["is_jfk"] == 1

    # Row 1: PULocationID 236 -> DOLocationID 236 -> is_same_zone == 1
    assert df_out.iloc[1]["is_same_zone"] == 1

    # Row 2: PULocationID 1 -> Newark -> is_newark == 1
    assert df_out.iloc[2]["is_newark"] == 1


def test_target_encoder_train_only_fit(sample_raw_dataframe):
    """Test target encoding fitting strictly on training set without leakage."""
    encoder = TargetCategoricalEncoder(smoothing=10.0)
    y_train = np.array([52.0, 8.5, 65.0])

    encoder.fit(sample_raw_dataframe, y_train)

    assert "fare" in encoder.global_means_
    assert "PULocationID" in encoder.target_maps_

    # Transform test set without fitting on test labels
    test_df = sample_raw_dataframe.copy()
    test_df["PULocationID"] = [132, 999, 1]  # 999 is unseen category
    df_transformed = encoder.transform(test_df)

    assert "PULocationID_target_enc" in df_transformed.columns
    # Unseen category 999 should safely fall back to global mean
    unseen_val = df_transformed.iloc[1]["PULocationID_target_enc"]
    assert unseen_val == encoder.global_means_["fare"]


def test_no_banned_columns_in_pipeline_output(sample_raw_dataframe):
    """Test leakage prevention: assert no banned post-trip column exists in pipeline output."""
    pipeline = NYCFeaturePipeline()
    y_train = np.array([[52.0, 25.0], [8.5, 5.0], [65.0, 35.0]])

    df_transformed = pipeline.fit_transform(sample_raw_dataframe, y_train)

    for banned in BANNED_COLUMNS:
        assert (
            banned not in df_transformed.columns
        ), f"Leakage detected: Banned column '{banned}' present in feature set!"


def test_feature_pipeline_serialization(sample_raw_dataframe, tmp_path):
    """Test serialization and deserialization of fitted NYCFeaturePipeline artifact."""
    pipeline = NYCFeaturePipeline()
    y_train = np.array([[52.0, 25.0], [8.5, 5.0], [65.0, 35.0]])

    pipeline.fit(sample_raw_dataframe, y_train)
    features_before = pipeline.transform(sample_raw_dataframe)

    # Save to temp pickle artifact
    pickle_file = tmp_path / "pipeline.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(pipeline, f)

    # Reload from pickle
    with open(pickle_file, "rb") as f:
        reloaded_pipeline = pickle.load(f)

    features_after = reloaded_pipeline.transform(sample_raw_dataframe)

    pd.testing.assert_frame_equal(features_before, features_after)


def test_pipeline_on_real_parquet_splits():
    """Integration test running NYCFeaturePipeline on actual train_cleaned.parquet."""
    if not os.path.exists(TRAIN_CLEANED_PATH):
        pytest.skip("train_cleaned.parquet not found")

    train_df = pd.read_parquet(TRAIN_CLEANED_PATH)
    X_train = train_df[ALLOWED_FEATURES].head(1000)
    y_train = train_df[["fare_amount", "duration_minutes"]].head(1000)

    pipeline = NYCFeaturePipeline()
    X_feat = pipeline.fit_transform(X_train, y_train)

    assert isinstance(X_feat, pd.DataFrame)
    assert len(X_feat) == 1000
    assert X_feat.shape[1] == 29
    assert "tpep_pickup_datetime" not in X_feat.columns
