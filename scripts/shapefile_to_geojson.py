# shapefile_to_geojson.py — Convierte el shapefile de zonas de NYC (T-116) a GeoJSON
# para el choropleth de T-111. Script one-off: se corre una sola vez, en el host, con
# geopandas (root requirements.txt) — el contenedor de ui/ nunca necesita geopandas/GDAL,
# solo lee el .geojson ya generado.

from pathlib import Path

import geopandas as gpd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
SHAPEFILE_SEARCH_DIR = DATASET_DIR / "taxi_zones"
OUTPUT_PATH = DATASET_DIR / "taxi_zones.geojson"


def find_shapefile() -> Path:
	candidates = list(SHAPEFILE_SEARCH_DIR.rglob("*.shp"))
	if not candidates:
		raise FileNotFoundError(
			f"No se encontró ningún .shp bajo {SHAPEFILE_SEARCH_DIR}. "
			"Corré la ingesta de T-116 primero (ver Tarea 8 del plan: python -m src.data_utils)."
		)
	return candidates[0]


def main() -> None:
	shp_path = find_shapefile()
	print(f"Leyendo shapefile desde {shp_path}...")
	gdf = gpd.read_file(shp_path)

	if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
		gdf = gdf.to_crs(epsg=4326)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	if OUTPUT_PATH.exists():
		OUTPUT_PATH.unlink()
	gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
	print(f"GeoJSON escrito en {OUTPUT_PATH} ({len(gdf)} zonas)")


if __name__ == "__main__":
	main()
