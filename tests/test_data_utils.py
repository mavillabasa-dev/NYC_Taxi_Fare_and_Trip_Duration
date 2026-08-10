# tests/test_data_utils.py — Unit tests for data ingestion, guardrails, and GIS centroids
import os
import pandas as pd
import pytest

from src.config import (
    MIN_LOOKUP_ROW_COUNT,
    MIN_PARQUET_ROW_COUNT,
    RAW_DATA_PATH,
    TAXI_ZONE_CENTROIDS_PATH,
    TAXI_ZONE_LOOKUP_PATH,
    TAXI_ZONE_SHAPEFILE_PATH,
)
from src.data_utils import (
    derive_zone_centroids,
    download_file,
    validate_file_size,
    validate_lookup_csv,
    validate_parquet_dataset,
)


def test_download_file_idempotency(tmp_path):
    """Test that download_file skips existing files that satisfy size guardrails."""
    test_file = tmp_path / "sample.txt"
    content = b"A" * 1000
    test_file.write_bytes(content)

    # Calling download_file with force=False should skip downloading
    downloaded = download_file(
        url="http://example.com/fake",
        target_path=str(test_file),
        min_expected_size=500,
        force=False,
    )
    assert downloaded is False


def test_download_file_truncation_assertion(tmp_path):
    """Test that validate_file_size raises ValueError if size is below minimum threshold."""
    test_file = tmp_path / "truncated.txt"
    test_file.write_bytes(b"short")

    with pytest.raises(ValueError, match="is below minimum expected threshold"):
        # Expect min 1000 bytes, but local file is only 5 bytes
        validate_file_size(str(test_file), min_expected_size=1000)


def test_validate_parquet_dataset():
    """Test parquet row count and size validation on actual ingested parquet file."""
    assert os.path.exists(RAW_DATA_PATH), "Raw parquet file should exist for test"
    rows = validate_parquet_dataset(RAW_DATA_PATH, min_row_count=MIN_PARQUET_ROW_COUNT)
    assert rows >= 3_000_000


def test_validate_lookup_csv():
    """Test zone lookup CSV validation."""
    assert os.path.exists(TAXI_ZONE_LOOKUP_PATH), "Lookup CSV should exist for test"
    rows = validate_lookup_csv(
        TAXI_ZONE_LOOKUP_PATH, min_row_count=MIN_LOOKUP_ROW_COUNT
    )
    assert rows >= 260


def test_derive_zone_centroids():
    """Test spatial centroid derivation from shapefile and lookup table."""
    assert os.path.exists(TAXI_ZONE_LOOKUP_PATH), "Lookup CSV should exist for test"

    centroids_df = derive_zone_centroids(
        shapefile_path=TAXI_ZONE_SHAPEFILE_PATH,
        lookup_path=TAXI_ZONE_LOOKUP_PATH,
        output_path=TAXI_ZONE_CENTROIDS_PATH,
        force=False,
    )

    assert isinstance(centroids_df, pd.DataFrame)
    expected_cols = {
        "LocationID",
        "Borough",
        "Zone",
        "service_zone",
        "longitude",
        "latitude",
    }
    assert expected_cols.issubset(centroids_df.columns)
    assert len(centroids_df) == 265

    # Check non-null centroid coordinates for LocationID 1 (EWR) and 132 (JFK)
    jfk = centroids_df[centroids_df["LocationID"] == 132].iloc[0]
    assert -74.0 <= jfk["longitude"] <= -73.0
    assert 40.0 <= jfk["latitude"] <= 41.0
