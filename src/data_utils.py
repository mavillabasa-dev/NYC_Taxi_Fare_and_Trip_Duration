# src/data_utils.py — Data loading, ingestion, and centroid derivation utilities
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    MIN_LOOKUP_ROW_COUNT,
    MIN_LOOKUP_SIZE_BYTES,
    MIN_PARQUET_ROW_COUNT,
    MIN_PARQUET_SIZE_BYTES,
    MIN_SHAPEFILE_SIZE_BYTES,
    RAW_DATA_PATH,
    TAXI_ZONE_CENTROIDS_PATH,
    TAXI_ZONE_LOOKUP_PATH,
    TAXI_ZONE_LOOKUP_URL,
    TAXI_ZONE_SHAPEFILE_PATH,
    TAXI_ZONE_SHAPEFILE_URL,
    YELLOW_TAXI_URL,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_file_size(target_path: str, min_expected_size: int) -> int:
    """
    Validates file size against minimum expected threshold.
    Raises ValueError if file size is truncated.
    """
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {target_path}")

    size = path.stat().st_size
    if size < min_expected_size:
        raise ValueError(
            f"Truncated download error: '{path.name}' size ({size} bytes) "
            f"is below minimum expected threshold ({min_expected_size} bytes)."
        )
    return size


def download_file(
    url: str, target_path: str, min_expected_size: int, force: bool = False
) -> bool:
    """
    Downloads a file from a URL to target_path idempotently.
    Skips download if file already exists and meets minimum size requirement.

    Args:
        url: Source URL string.
        target_path: Destination local file path.
        min_expected_size: Minimum required size in bytes to prevent truncated downloads.
        force: If True, re-downloads even if target file exists.

    Returns:
        True if file was downloaded, False if skipped because it already exists.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and target.exists() and target.stat().st_size >= min_expected_size:
        logger.info(
            f"Skipping download for '{target.name}' — already present ({target.stat().st_size / (1024*1024):.2f} MB)"
        )
        return False

    logger.info(f"Downloading '{target.name}' from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(target, "wb") as out_file:
        out_file.write(response.read())

    validate_file_size(target_path, min_expected_size)
    logger.info(
        f"Downloaded '{target.name}' ({target.stat().st_size / (1024*1024):.2f} MB)"
    )
    return True


def validate_parquet_dataset(
    parquet_path: str, min_row_count: int = MIN_PARQUET_ROW_COUNT
) -> int:
    """
    Validates parquet file integrity by checking file existence, size, and row count.

    Args:
        parquet_path: Path to raw parquet file.
        min_row_count: Minimum expected row count.

    Returns:
        Row count integer.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}")

    file_size = path.stat().st_size
    if file_size < MIN_PARQUET_SIZE_BYTES:
        raise ValueError(
            f"Parquet file size ({file_size} bytes) is below minimum threshold ({MIN_PARQUET_SIZE_BYTES} bytes)"
        )

    parquet_file = pq.ParquetFile(parquet_path)
    row_count = parquet_file.metadata.num_rows
    logger.info(
        f"Verified Parquet '{path.name}': {row_count:,} rows, {file_size / (1024*1024):.2f} MB"
    )

    if row_count < min_row_count:
        raise ValueError(
            f"Parquet row count ({row_count:,}) is below minimum expected ({min_row_count:,})"
        )

    return row_count


def validate_lookup_csv(
    csv_path: str, min_row_count: int = MIN_LOOKUP_ROW_COUNT
) -> int:
    """
    Validates Taxi Zone Lookup CSV integrity.

    Args:
        csv_path: Path to zone lookup CSV file.
        min_row_count: Minimum expected zone row count.

    Returns:
        Row count integer.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Lookup CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"LocationID", "Borough", "Zone", "service_zone"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Lookup CSV is missing required columns: {missing}")

    row_count = len(df)
    logger.info(f"Verified Lookup CSV '{path.name}': {row_count} zones")

    if row_count < min_row_count:
        raise ValueError(
            f"Lookup CSV row count ({row_count}) is below minimum expected ({min_row_count})"
        )

    return row_count


def derive_zone_centroids(
    shapefile_path: str = TAXI_ZONE_SHAPEFILE_PATH,
    lookup_path: str = TAXI_ZONE_LOOKUP_PATH,
    output_path: str = TAXI_ZONE_CENTROIDS_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """
    Derives spatial centroids (longitude, latitude in WGS84 EPSG:4326) from Taxi Zone Shapefile,
    merges them with Zone Lookup metadata, and caches the result as a lookup table keyed by LocationID.

    Args:
        shapefile_path: Path to local taxi_zones.zip.
        lookup_path: Path to local taxi_zone_lookup.csv.
        output_path: Path to save output cached lookup CSV.
        force: If True, re-derives centroids even if output file exists.

    Returns:
        Merged DataFrame containing LocationID, Borough, Zone, service_zone, longitude, latitude.
    """
    out_p = Path(output_path)
    if not force and out_p.exists() and out_p.stat().st_size > 0:
        logger.info(f"Loading cached zone centroids from {out_p.name}...")
        centroids_df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(centroids_df)} zone centroids from cache.")
        return centroids_df

    logger.info(f"Extracting and reading shapefile from {shapefile_path}...")
    shapefile_dir = Path(shapefile_path).parent / "taxi_zones"
    shapefile_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
        zip_ref.extractall(shapefile_dir)

    shp_files = list(shapefile_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"No .shp file found inside extracted shapefile directory {shapefile_dir}"
        )

    shp_path = shp_files[0]
    gdf = gpd.read_file(shp_path)

    # Compute centroids in native projected CRS (e.g. EPSG:2263 feet), then reproject centroids to WGS84 (EPSG:4326)
    if gdf.crs is not None:
        centroids_native = gdf.geometry.centroid
        centroids_wgs84 = gpd.GeoSeries(centroids_native, crs=gdf.crs).to_crs(epsg=4326)
    else:
        centroids_wgs84 = gdf.to_crs(epsg=4326).geometry.centroid

    df_shape_centroids = pd.DataFrame(
        {
            "LocationID": gdf["LocationID"],
            "longitude": centroids_wgs84.x,
            "latitude": centroids_wgs84.y,
        }
    )

    # Load lookup table
    df_lookup = pd.read_csv(lookup_path)

    # Merge shapefile centroids with lookup table
    merged_df = pd.merge(df_lookup, df_shape_centroids, on="LocationID", how="left")
    merged_df = merged_df.sort_values("LocationID").reset_index(drop=True)

    # Save cached lookup table
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(
        f"Successfully derived and saved {len(merged_df)} zone centroids to {output_path}"
    )

    return merged_df


def ingest_all_data(force: bool = False) -> Dict[str, Any]:
    """
    One-command reproducible ingestion pipeline:
    1. Downloads parquet dataset, zone lookup CSV, and zone shapefile ZIP idempotently.
    2. Validates sizes and row counts to prevent silent truncation.
    3. Derives and caches spatial zone centroids lookup table.

    Args:
        force: If True, re-downloads and re-computes all artifacts.

    Returns:
        Metadata dictionary summarizing ingestion status and row counts.
    """
    logger.info("--- Starting Reproducible Dataset Ingestion (T-116) ---")

    # 1. Download files idempotently
    download_file(YELLOW_TAXI_URL, RAW_DATA_PATH, MIN_PARQUET_SIZE_BYTES, force=force)
    download_file(
        TAXI_ZONE_LOOKUP_URL, TAXI_ZONE_LOOKUP_PATH, MIN_LOOKUP_SIZE_BYTES, force=force
    )
    download_file(
        TAXI_ZONE_SHAPEFILE_URL,
        TAXI_ZONE_SHAPEFILE_PATH,
        MIN_SHAPEFILE_SIZE_BYTES,
        force=force,
    )

    # 2. Validate data files
    parquet_rows = validate_parquet_dataset(RAW_DATA_PATH)
    lookup_rows = validate_lookup_csv(TAXI_ZONE_LOOKUP_PATH)

    # 3. Derive spatial centroids
    centroids_df = derive_zone_centroids(
        shapefile_path=TAXI_ZONE_SHAPEFILE_PATH,
        lookup_path=TAXI_ZONE_LOOKUP_PATH,
        output_path=TAXI_ZONE_CENTROIDS_PATH,
        force=force,
    )

    logger.info("--- Dataset Ingestion Completed Successfully ---")
    return {
        "parquet_rows": parquet_rows,
        "lookup_rows": lookup_rows,
        "centroids_count": len(centroids_df),
        "status": "success",
    }


if __name__ == "__main__":
    ingest_all_data()
