"""Configuración compartida para recolectar los dos árboles de tests."""

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
    """Registrar marcadores personalizados de pytest."""
    config.addinivalue_line(
        "markers",
        "requires_dataset: marca tests que requieren los archivos reales en dataset/",
    )


def pytest_collection_modifyitems(config, items):
    """Omitir pruebas que necesitan datos locales no descargados."""
    dataset_dir = ROOT_DIR / "dataset"
    parquet_exists = (dataset_dir / "yellow_tripdata_2022-05.parquet").exists()
    lookup_exists = (dataset_dir / "taxi_zone_lookup.csv").exists()
    shapefile_exists = (dataset_dir / "taxi_zones.zip").exists()

    if parquet_exists and lookup_exists and shapefile_exists:
        return

    for item in items:
        nodeid = item.nodeid.lower()
        if any(
            keyword in nodeid
            for keyword in (
                "test_validate_parquet_dataset",
                "test_validate_lookup_csv",
                "test_derive_zone_centroids",
            )
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Dataset local no disponible; descarga/ingesta antes de ejecutar estas pruebas."
                )
            )