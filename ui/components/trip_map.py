"""Mapa de puntos: muestra el pickup y el dropoff del viaje predicho sobre NYC."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from zones import load_zones


def render(pu_location_id: int, do_location_id: int) -> None:
	zones = load_zones()
	pu_row = zones[zones.LocationID == pu_location_id]
	do_row = zones[zones.LocationID == do_location_id]

	if pu_row.empty or do_row.empty:
		st.info("No hay datos de zona para mostrar en el mapa.")
		return

	pu_row = pu_row.iloc[0]
	do_row = do_row.iloc[0]

	if pd.isna(pu_row.longitude) or pd.isna(do_row.longitude):
		st.info(
			"Una de las zonas elegidas no tiene coordenadas conocidas "
			"(p. ej. 'Outside of NYC'), no se puede dibujar en el mapa."
		)
		return

	fig = go.Figure(
		go.Scattermapbox(
			lat=[pu_row.latitude, do_row.latitude],
			lon=[pu_row.longitude, do_row.longitude],
			mode="markers+text",
			text=["Pickup", "Dropoff"],
			textposition="top center",
			marker=dict(size=14, color=["#2ecc71", "#e74c3c"]),
		)
	)
	fig.update_layout(
		mapbox_style="carto-positron",
		mapbox_zoom=10,
		mapbox_center={
			"lat": (pu_row.latitude + do_row.latitude) / 2,
			"lon": (pu_row.longitude + do_row.longitude) / 2,
		},
		margin=dict(l=0, r=0, t=0, b=0),
		height=400,
	)
	st.plotly_chart(fig, use_container_width=True)
