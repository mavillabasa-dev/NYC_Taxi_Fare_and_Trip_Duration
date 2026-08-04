# src/preprocessing.py — Data cleaning, filtering, and preprocessing module
import logging
import os
import pandas as pd
from typing import Dict, Tuple

from src.config import (
    ALLOWED_FEATURES,
    BANNED_COLUMNS,
    MAX_DURATION_MINUTES,
    MAX_FARE_AMOUNT,
    MAX_LOCATION_ID,
    MAX_PASSENGER_COUNT,
    MAX_TRIP_DISTANCE,
    MAY_2022_END,
    MAY_2022_START,
    MIN_DURATION_MINUTES,
    MIN_FARE_AMOUNT,
    MIN_LOCATION_ID,
    MIN_PASSENGER_COUNT,
    MIN_TRIP_DISTANCE,
    RAW_DATA_PATH,
    TARGET_COLUMNS,
    TEMPORAL_SPLIT_DATE,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
    VALID_RATECODES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates trip duration in minutes from pickup and dropoff datetimes.

    Args:
        df: Raw or partially processed DataFrame containing pickup/dropoff datetimes.

    Returns:
        DataFrame with an added 'duration_minutes' float column.
    """
    df = df.copy()
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60.0
    return df


def filter_outliers(
    df: pd.DataFrame, verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Applies pure data cleaning and outlier filtering rules sequentially.
    Tracks and logs the exact count of rows dropped by each rule.

    Args:
        df: Input DataFrame with duration_minutes already calculated.
        verbose: Whether to log cleaning stats to stdout.

    Returns:
        Tuple of (cleaned_df, cleaning_stats_dict).
    """
    cleaned_df = df.copy()
    initial_count = len(cleaned_df)
    stats: Dict[str, int] = {}

    # Rule 1: Pickup timestamp within May 2022
    mask_time = (cleaned_df["tpep_pickup_datetime"] >= MAY_2022_START) & (
        cleaned_df["tpep_pickup_datetime"] < MAY_2022_END
    )
    dropped_time = (~mask_time).sum()
    cleaned_df = cleaned_df[mask_time]
    stats["out_of_bound_timestamps"] = int(dropped_time)

    # Rule 2: Duration within valid bounds (0.5 to 360 min)
    mask_dur = (cleaned_df["duration_minutes"] >= MIN_DURATION_MINUTES) & (
        cleaned_df["duration_minutes"] <= MAX_DURATION_MINUTES
    )
    dropped_dur = (~mask_dur).sum()
    cleaned_df = cleaned_df[mask_dur]
    stats["invalid_durations"] = int(dropped_dur)

    # Rule 3: Fare amount within valid bounds ($2.50 to $500.00)
    mask_fare = (cleaned_df["fare_amount"] >= MIN_FARE_AMOUNT) & (
        cleaned_df["fare_amount"] <= MAX_FARE_AMOUNT
    )
    dropped_fare = (~mask_fare).sum()
    cleaned_df = cleaned_df[mask_fare]
    stats["invalid_fares"] = int(dropped_fare)

    # Rule 4: Trip distance within valid bounds (>0 to 150 miles)
    mask_dist = (cleaned_df["trip_distance"] >= MIN_TRIP_DISTANCE) & (
        cleaned_df["trip_distance"] <= MAX_TRIP_DISTANCE
    )
    dropped_dist = (~mask_dist).sum()
    cleaned_df = cleaned_df[mask_dist]
    stats["invalid_trip_distances"] = int(dropped_dist)

    # Rule 5: Valid passenger count (non-null, between 1 and 9)
    mask_pass = (
        cleaned_df["passenger_count"].notnull()
        & (cleaned_df["passenger_count"] >= MIN_PASSENGER_COUNT)
        & (cleaned_df["passenger_count"] <= MAX_PASSENGER_COUNT)
    )
    dropped_pass = (~mask_pass).sum()
    cleaned_df = cleaned_df[mask_pass]
    stats["invalid_passenger_counts"] = int(dropped_pass)

    # Rule 6: Valid RatecodeID (non-null, in 1..6)
    mask_rate = cleaned_df["RatecodeID"].notnull() & cleaned_df["RatecodeID"].isin(
        VALID_RATECODES
    )
    dropped_rate = (~mask_rate).sum()
    cleaned_df = cleaned_df[mask_rate]
    stats["invalid_ratecodes"] = int(dropped_rate)

    # Rule 7: Valid Location IDs (1 to 265)
    mask_loc = (
        (cleaned_df["PULocationID"] >= MIN_LOCATION_ID)
        & (cleaned_df["PULocationID"] <= MAX_LOCATION_ID)
        & (cleaned_df["DOLocationID"] >= MIN_LOCATION_ID)
        & (cleaned_df["DOLocationID"] <= MAX_LOCATION_ID)
    )
    dropped_loc = (~mask_loc).sum()
    cleaned_df = cleaned_df[mask_loc]
    stats["invalid_location_ids"] = int(dropped_loc)

    final_count = len(cleaned_df)
    total_dropped = initial_count - final_count
    stats["total_initial_rows"] = initial_count
    stats["total_final_rows"] = final_count
    stats["total_dropped_rows"] = total_dropped
    stats["retention_rate_pct"] = round((final_count / initial_count) * 100, 2)

    if verbose:
        logger.info(f"Initial raw rows: {initial_count:,}")
        for rule, count in stats.items():
            if rule not in [
                "total_initial_rows",
                "total_final_rows",
                "total_dropped_rows",
                "retention_rate_pct",
            ]:
                logger.info(f"  - Dropped by {rule}: {count:,}")
        logger.info(
            f"Total rows dropped: {total_dropped:,} ({100 - stats['retention_rate_pct']}%)"
        )
        logger.info(
            f"Final clean rows: {final_count:,} ({stats['retention_rate_pct']}%)"
        )

    return cleaned_df, stats


def drop_banned_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all post-trip leakage columns in a single explicit step.
    Ensures ONLY allowed prediction-time features and target columns survive.

    Args:
        df: Filtered DataFrame containing raw columns.

    Returns:
        DataFrame containing ONLY ALLOWED_FEATURES + TARGET_COLUMNS.
    """
    # Verify no unexpected columns
    expected_cols = ALLOWED_FEATURES + TARGET_COLUMNS

    # Drop any banned columns present
    cols_to_drop = [col for col in df.columns if col in BANNED_COLUMNS]
    cleaned = df.drop(columns=cols_to_drop, errors="ignore")

    # Keep strictly the allowed features + targets
    final_cols = [col for col in expected_cols if col in cleaned.columns]
    cleaned = cleaned[final_cols]

    # Assert leakage prevention
    for banned in BANNED_COLUMNS:
        assert (
            banned not in cleaned.columns
        ), f"LEAKAGE ERROR: Banned column '{banned}' survived!"

    return cleaned


def perform_temporal_split(
    df: pd.DataFrame, split_date: str = TEMPORAL_SPLIT_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict temporal train/test split (not random) based on pickup datetime.

    Args:
        df: Preprocessed DataFrame containing 'tpep_pickup_datetime'.
        split_date: Datetime string constant serving as the split boundary.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.copy()
    split_dt = pd.to_datetime(split_date)
    pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])

    train_df = df[pickup_dt < split_dt].reset_index(drop=True)
    test_df = df[pickup_dt >= split_dt].reset_index(drop=True)

    logger.info(f"Temporal Split at {split_date}:")
    logger.info(f"  - Train Set (Pre-{split_date}): {len(train_df):,} rows")
    logger.info(f"  - Test Set (Post-{split_date}): {len(test_df):,} rows")

    return train_df, test_df


def clean_and_preprocess_dataset(
    input_path: str = RAW_DATA_PATH, save_output: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Main orchestrator for data cleaning, leakage removal, and temporal splitting.

    Args:
        input_path: Path to raw yellow taxi parquet file.
        save_output: Whether to write train_cleaned.parquet and test_cleaned.parquet to dataset/.

    Returns:
        Tuple of (train_df, test_df, cleaning_stats_dict).
    """
    logger.info(f"Loading raw dataset from {input_path}...")
    raw_df = pd.read_parquet(input_path)

    # 1. Calculate duration
    df_with_duration = calculate_trip_duration(raw_df)

    # 2. Filter outliers
    filtered_df, stats = filter_outliers(df_with_duration, verbose=True)

    # 3. Drop banned leakage columns
    feature_df = drop_banned_columns(filtered_df)

    # 4. Perform temporal split
    train_df, test_df = perform_temporal_split(feature_df, TEMPORAL_SPLIT_DATE)

    # 5. Save parquet files if requested
    if save_output:
        os.makedirs(os.path.dirname(TRAIN_CLEANED_PATH), exist_ok=True)
        train_df.to_parquet(TRAIN_CLEANED_PATH, index=False)
        test_df.to_parquet(TEST_CLEANED_PATH, index=False)
        logger.info(
            f"Saved cleaned train split to {TRAIN_CLEANED_PATH} ({len(train_df):,} rows)"
        )
        logger.info(
            f"Saved cleaned test split to {TEST_CLEANED_PATH} ({len(test_df):,} rows)"
        )

    return train_df, test_df, stats


if __name__ == "__main__":
    clean_and_preprocess_dataset()
