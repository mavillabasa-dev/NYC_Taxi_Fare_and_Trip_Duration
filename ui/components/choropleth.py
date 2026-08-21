"""Choropleth: tarifa predicha desde la zona de pickup elegida hacia el resto de NYC.

Nota de aproximación: no existe una distancia de ruteo real para pares de zonas que
todavía no fueron viajados, así que se usa la distancia haversine entre centroides como
proxy de `trip_distance` en cada llamada a /predict — mismo tipo de aproximación
documentada para `trip_distance` en el resto del proyecto (ver README, sección Dataset).
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
from settings import settings
from zones import (
	MAX_TRIP_DISTANCE_MILES,
	MIN_TRIP_DISTANCE_MILES,
	haversine_miles,
	infer_ratecode,
	load_zones,
)


@st.cache_data
def _load_geojson() -> dict:
	geojson_path = Path(settings.dataset_dir) / "taxi_zones.geojson"
	with geojson_path.open() as f:
		return json.load(f)


@st.cache_data(show_spinner=False)
def _predict_grid(pu_location_id: int, pickup_dt: str, passengers: int) -> pd.DataFrame:
	zones = load_zones().dropna(subset=["longitude", "latitude"])
	pu_rows = zones[zones.LocationID == pu_location_id]
	if pu_rows.empty:
		return pd.DataFrame(columns=["LocationID", "predicted_fare"])
	pu_row = pu_rows.iloc[0]

	rows = []
	total = len(zones)
	progress = st.progress(0.0, text="Calculando tarifas por zona...")
	for i, (_, do_row) in enumerate(zones.iterrows()):
		distance = haversine_miles(pu_row.latitude, pu_row.longitude, do_row.latitude, do_row.longitude)
		distance = min(max(distance, MIN_TRIP_DISTANCE_MILES), MAX_TRIP_DISTANCE_MILES)
		# El rate code se infiere por cada par (pickup, dropoff) — no es un valor
		# fijo para las 263 filas, igual que en el formulario principal.
		do_location_id = int(do_row.LocationID)
		payload = {
			"PULocationID": int(pu_location_id),
			"DOLocationID": do_location_id,
			"tpep_pickup_datetime": pickup_dt,
			"passenger_count": passengers,
			"RatecodeID": infer_ratecode(pu_location_id, do_location_id),
			"trip_distance": round(distance, 2),
		}
		status_code, body = api_client.predict(payload)
		if status_code == 200:
			rows.append({"LocationID": do_location_id, "predicted_fare": body["predicted_fare"]})
		progress.progress((i + 1) / total, text=f"Calculando tarifas por zona... ({i + 1}/{total})")
	progress.empty()
	return pd.DataFrame(rows)


def render(pu_location_id: int, pickup_dt: str, passengers: int) -> None:
	grid = _predict_grid(pu_location_id, pickup_dt, passengers)

	if grid.empty:
		st.info("No se pudieron calcular tarifas para el choropleth (zona de pickup sin coordenadas).")
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
