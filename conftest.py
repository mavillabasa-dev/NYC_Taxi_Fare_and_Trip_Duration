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

DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"

SKIP_DATASET_REASON = (
    "dataset/ is not populated. Run `python -m src.data_utils` to ingest the TLC "
    "files before running these tests."
)
SKIP_MODEL_REASON = (
    "models/model.pkl not found. Run `python scripts/dev_fixture_model.py` for a "
    "quick stub artifact, or `python -m scripts.run_training_pipeline` for the real "
    "trained model."
)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_dataset: marks tests that require the ingested TLC files in dataset/",
    )
    config.addinivalue_line(
        "markers",
        "requires_model: marks tests that require a serialized bundle at models/model.pkl",
    )


def _dataset_is_available() -> bool:
    """Reports the dataset as it looks *after* a successful ingestion.

    Deliberately does not look for dataset/taxi_zones.zip. derive_zone_centroids
    deletes that archive once it has extracted it, so gating on the zip would skip
    these tests forever after the very ingestion meant to enable them.
    """
    if not (DATASET_DIR / "yellow_tripdata_2022-05.parquet").exists():
        return False
    if not (DATASET_DIR / "taxi_zone_lookup.csv").exists():
        return False

    # derive_zone_centroids(force=False) is satisfied by either the cached lookup
    # table or the extracted shapefile it would otherwise be rebuilt from.
    shapefile_dir = DATASET_DIR / "taxi_zones"
    extracted = shapefile_dir.is_dir() and any(shapefile_dir.rglob("*.shp"))
    return (DATASET_DIR / "taxi_zone_centroids.csv").exists() or extracted


def _model_is_available() -> bool:
    return (MODELS_DIR / "model.pkl").exists()


def pytest_collection_modifyitems(config, items):
    """Skip tests whose local inputs have not been produced yet.

    Both dataset/ and models/ are gitignored and built by scripts, so a clean
    checkout legitimately has neither. Marked tests skip with instructions instead
    of failing.
    """
    gates = []
    if not _dataset_is_available():
        gates.append(("requires_dataset", pytest.mark.skip(reason=SKIP_DATASET_REASON)))
    if not _model_is_available():
        gates.append(("requires_model", pytest.mark.skip(reason=SKIP_MODEL_REASON)))

    if not gates:
        return

    for item in items:
        for marker_name, skip_marker in gates:
            if item.get_closest_marker(marker_name) is not None:
                item.add_marker(skip_marker)
