"""Carga y cachea el catálogo de zonas de NYC (nombre, borough, coordenadas)."""

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from settings import settings

JFK_LOCATION_ID = 132
NEWARK_LOCATION_ID = 1

RATECODE_LABELS = {
	1: "Standard",
	2: "JFK",
	3: "Newark",
	4: "Nassau/Westchester",
	5: "Negotiated fare",
	6: "Group ride",
}

EARTH_RADIUS_MILES = 3958.8
MIN_TRIP_DISTANCE_MILES = 0.1
MAX_TRIP_DISTANCE_MILES = 150.0
# Respaldo cuando una zona no tiene coordenadas conocidas (LocationID 264/265) — mismo
# default que tenía el campo manual antes de auto-completarse.
DEFAULT_TRIP_DISTANCE_MILES = 3.0


@st.cache_data
def load_zones() -> pd.DataFrame:
	centroids_path = Path(settings.dataset_dir) / "taxi_zone_centroids.csv"
	df = pd.read_csv(centroids_path)
	df["Borough"] = df["Borough"].fillna("Unknown")
	df["Zone"] = df["Zone"].fillna("Unknown")
	df["label"] = df["Zone"] + " (" + df["Borough"] + ")"
	return df.sort_values("label").reset_index(drop=True)


def infer_ratecode(pu_location_id: int, do_location_id: int) -> int:
	if JFK_LOCATION_ID in (pu_location_id, do_location_id):
		return 2
	if NEWARK_LOCATION_ID in (pu_location_id, do_location_id):
		return 3
	return 1


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
	return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def estimate_trip_distance(pu_location_id: int, do_location_id: int) -> float:
	"""Aproxima trip_distance con la distancia haversine (línea recta) entre pickup y
	dropoff. Es una subestimación sistemática frente a una ruta real en auto — sirve
	como mejora incremental sobre un input manual, no reemplaza una API de ruteo."""
	zones = load_zones()
	pu_rows = zones[zones.LocationID == pu_location_id]
	do_rows = zones[zones.LocationID == do_location_id]
	if pu_rows.empty or do_rows.empty:
		return DEFAULT_TRIP_DISTANCE_MILES

	pu_row, do_row = pu_rows.iloc[0], do_rows.iloc[0]
	if pd.isna(pu_row.longitude) or pd.isna(do_row.longitude):
		return DEFAULT_TRIP_DISTANCE_MILES

	distance = haversine_miles(pu_row.latitude, pu_row.longitude, do_row.latitude, do_row.longitude)
	return round(min(max(distance, MIN_TRIP_DISTANCE_MILES), MAX_TRIP_DISTANCE_MILES), 2)
