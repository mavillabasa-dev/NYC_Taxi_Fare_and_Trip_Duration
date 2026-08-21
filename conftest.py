"""Shared configuration to collect from both test trees."""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent
API_DIR = ROOT_DIR / "api"

for path in (ROOT_DIR, API_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_dataset: marks tests that require actual dataset files in dataset/",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require local dataset files if not downloaded."""
    dataset_dir = ROOT_DIR / "dataset"
    parquet_exists = (dataset_dir / "yellow_tripdata_2022-05.parquet").exists()
    lookup_exists = (dataset_dir / "taxi_zone_lookup.csv").exists()
    shapefile_exists = (dataset_dir / "taxi_zones.zip").exists()

    if parquet_exists and lookup_exists and shapefile_exists:
        return

    for item in items:
        has_marker = item.get_closest_marker("requires_dataset") is not None
        nodeid = item.nodeid.lower()
        matches_name = any(
            keyword in nodeid
            for keyword in (
                "test_validate_parquet_dataset",
                "test_validate_lookup_csv",
                "test_derive_zone_centroids",
            )
        )
        if has_marker or matches_name:
            item.add_marker(
                pytest.mark.skip(
                    reason="Local dataset not available; download/ingest before running these tests."
                )
            )