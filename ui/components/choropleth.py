"""Choropleth: Predicted fare from the selected pickup zone to the rest of NYC.

Approximation note: No actual road routing distance exists for unvisited zone pairs,
so Haversine centroid distance is used as a proxy for `trip_distance` in each /predict call —
the same approximation documented for `trip_distance` throughout the project (see README, Dataset section).
"""

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
from settings import settings
from zones import load_zones

EARTH_RADIUS_MILES = 3958.8


@st.cache_data
def _load_geojson() -> dict:
	geojson_path = Path(settings.dataset_dir) / "taxi_zones.geojson"
	with geojson_path.open() as f:
		return json.load(f)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
	return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


@st.cache_data(show_spinner=False)
def _predict_grid(
	pu_location_id: int, pickup_dt: str, passengers: int, ratecode: int
) -> pd.DataFrame:
	zones = load_zones().dropna(subset=["longitude", "latitude"])
	pu_rows = zones[zones.LocationID == pu_location_id]
	if pu_rows.empty:
		return pd.DataFrame(columns=["LocationID", "predicted_fare"])
	pu_row = pu_rows.iloc[0]

	rows = []
	total = len(zones)
	progress = st.progress(0.0, text="Calculating fares across zones...")
	for i, (_, do_row) in enumerate(zones.iterrows()):
		distance = _haversine_miles(pu_row.latitude, pu_row.longitude, do_row.latitude, do_row.longitude)
		distance = min(max(distance, 0.1), 150.0)
		payload = {
			"PULocationID": int(pu_location_id),
			"DOLocationID": int(do_row.LocationID),
			"tpep_pickup_datetime": pickup_dt,
			"passenger_count": passengers,
			"RatecodeID": ratecode,
			"trip_distance": round(distance, 2),
		}
		status_code, body = api_client.predict(payload)
		if status_code == 200:
			rows.append({"LocationID": int(do_row.LocationID), "predicted_fare": body["predicted_fare"]})
		progress.progress((i + 1) / total, text=f"Calculating fares across zones... ({i + 1}/{total})")
	progress.empty()
	return pd.DataFrame(rows)


def render(pu_location_id: int, pickup_dt: str, passengers: int, ratecode: int) -> None:
	grid = _predict_grid(pu_location_id, pickup_dt, passengers, ratecode)

	if grid.empty:
		st.info("Could not calculate fares for choropleth map (pickup zone lacks coordinates).")
		return

	fig = px.choropleth_mapbox(
		grid,
		geojson=_load_geojson(),
		locations="LocationID",
		featureidkey="properties.LocationID",
		color="predicted_fare",
		color_continuous_scale="YlOrRd",
		mapbox_style="carto-positron",
		zoom=9,
		center={"lat": 40.75, "lon": -73.95},
		opacity=0.7,
		labels={"predicted_fare": "Predicted fare ($)"},
	)
	fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
	st.plotly_chart(fig, use_container_width=True)
