"""Point map: displays pickup and dropoff of the predicted trip on NYC."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import (
	DEFAULT_POINT_MAP_ZOOM,
	DROPOFF_MARKER_COLOR,
	MARKER_SIZE,
	PICKUP_MARKER_COLOR,
	POINT_MAP_HEIGHT,
)
from zones import load_zones


def render(pu_location_id: int, do_location_id: int) -> None:
	zones = load_zones()
	pu_row = zones[zones.LocationID == pu_location_id]
	do_row = zones[zones.LocationID == do_location_id]

	if pu_row.empty or do_row.empty:
		st.info("No zone data available to display on map.")
		return

	pu_row = pu_row.iloc[0]
	do_row = do_row.iloc[0]

	if pd.isna(pu_row.longitude) or pd.isna(do_row.longitude):
		st.info(
			"One of the selected zones does not have known GPS coordinates "
			"(e.g. 'Outside of NYC') and cannot be rendered on the map."
		)
		return

	fig = go.Figure(
		go.Scattermapbox(
			lat=[pu_row.latitude, do_row.latitude],
			lon=[pu_row.longitude, do_row.longitude],
			mode="markers+text",
			text=["Pickup", "Dropoff"],
			textposition="top center",
			marker=dict(size=MARKER_SIZE, color=[PICKUP_MARKER_COLOR, DROPOFF_MARKER_COLOR]),
			# Without this, Plotly displays raw lat/lon in hover by default.
			# hoverinfo="text" instructs it to use hovertext instead.
			hovertext=[f"Pickup: {pu_row.label}", f"Dropoff: {do_row.label}"],
			hoverinfo="text",
		)
	)
	fig.update_layout(
		mapbox_style="carto-positron",
		mapbox_zoom=DEFAULT_POINT_MAP_ZOOM,
		mapbox_center={
			"lat": (pu_row.latitude + do_row.latitude) / 2,
			"lon": (pu_row.longitude + do_row.longitude) / 2,
		},
		margin=dict(l=0, r=0, t=0, b=0),
		height=POINT_MAP_HEIGHT,
	)
	st.plotly_chart(fig, use_container_width=True)
