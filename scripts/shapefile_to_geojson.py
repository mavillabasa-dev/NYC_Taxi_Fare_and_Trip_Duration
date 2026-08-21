# shapefile_to_geojson.py — Converts NYC taxi zones shapefile (T-116) to GeoJSON
# for the T-111 choropleth. One-off script: runs once on the host with
# geopandas (root requirements.txt) — the ui/ container never needs geopandas/GDAL,
# it only reads the pre-generated .geojson.

from pathlib import Path

import geopandas as gpd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
SHAPEFILE_SEARCH_DIR = DATASET_DIR / "taxi_zones"
OUTPUT_PATH = DATASET_DIR / "taxi_zones.geojson"


def find_shapefile() -> Path:
	candidates = list(SHAPEFILE_SEARCH_DIR.rglob("*.shp"))
	if not candidates:
		raise FileNotFoundError(
			f"No .shp found under {SHAPEFILE_SEARCH_DIR}. "
			"Run T-116 data ingestion first (python -m src.data_utils)."
		)
	return candidates[0]


def main() -> None:
	shp_path = find_shapefile()
	print(f"Reading shapefile from {shp_path}...")
	gdf = gpd.read_file(shp_path)

	if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
		gdf = gdf.to_crs(epsg=4326)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	if OUTPUT_PATH.exists():
		OUTPUT_PATH.unlink()
	gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
	print(f"GeoJSON written to {OUTPUT_PATH} ({len(gdf)} zones)")


if __name__ == "__main__":
	main()
