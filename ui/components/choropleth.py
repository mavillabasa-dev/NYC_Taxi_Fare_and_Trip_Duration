"""Choropleth: Predicted fare from the selected pickup zone to the rest of NYC.

Approximation note: No actual road routing distance exists for unvisited zone pairs,
so Haversine centroid distance is used as a proxy for `trip_distance` in each /predict call —
the same approximation documented for `trip_distance` throughout the project (see README, Dataset section).
"""

import json
from pathlib import Path

import api_client
import pandas as pd
import plotly.express as px
import streamlit as st
from settings import (
    CHOROPLETH_HEIGHT,
    DEFAULT_CHOROPLETH_ZOOM,
    DEFAULT_MAP_CENTER,
    DEFAULT_MAP_OPACITY,
    FARE_COLOR_SCALE,
    HTTP_STATUS_OK,
    MAP_FONT,
    MAP_MUTED_COLOR,
    MAP_STYLE,
    MAP_TEXT_COLOR,
    TRIP_DISTANCE_MAX,
    TRIP_DISTANCE_MIN,
    settings,
)
from zones import haversine_miles, infer_ratecode, load_zones


@st.cache_data
def _load_geojson() -> dict:
    geojson_path = Path(settings.dataset_dir) / "taxi_zones.geojson"
    with geojson_path.open() as f:
        return json.load(f)


def _zone_label(location_id: int) -> str:
    """Human-readable name for a zone, for captions."""
    rows = load_zones()
    match = rows[rows.LocationID == location_id]
    return str(match.iloc[0].label) if not match.empty else f"zone {location_id}"


@st.cache_data(show_spinner=False)
def _predict_grid(pu_location_id: int, pickup_dt: str, passengers: int) -> pd.DataFrame:
    zones = load_zones().dropna(subset=["longitude", "latitude"])
    pu_rows = zones[zones.LocationID == pu_location_id]
    if pu_rows.empty:
        return pd.DataFrame(columns=["LocationID", "predicted_fare"])
    pu_row = pu_rows.iloc[0]

    rows = []
    total = len(zones)
    progress = st.progress(0.0, text="Calculating fares across zones...")
    for i, (_, do_row) in enumerate(zones.iterrows()):
        distance = haversine_miles(
            pu_row.latitude, pu_row.longitude, do_row.latitude, do_row.longitude
        )
        distance = min(max(distance, TRIP_DISTANCE_MIN), TRIP_DISTANCE_MAX)
        # Rate code is inferred per (pickup, dropoff) pair — not a fixed
        # value across all rows, matching the main form.
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
        if status_code == HTTP_STATUS_OK:
            rows.append({"LocationID": do_location_id, "predicted_fare": body["predicted_fare"]})
        progress.progress(
            (i + 1) / total, text=f"Calculating fares across zones... ({i + 1}/{total})"
        )
    progress.empty()
    return pd.DataFrame(rows)


def render(pu_location_id: int, pickup_dt: str, passengers: int) -> None:
    grid = _predict_grid(pu_location_id, pickup_dt, passengers)

    if grid.empty:
        st.info("Could not calculate fares for choropleth map (pickup zone lacks coordinates).")
        return

    fig = px.choropleth_mapbox(
        grid,
        geojson=_load_geojson(),
        locations="LocationID",
        featureidkey="properties.LocationID",
        color="predicted_fare",
        color_continuous_scale=FARE_COLOR_SCALE,
        mapbox_style=MAP_STYLE,
        zoom=DEFAULT_CHOROPLETH_ZOOM,
        center=DEFAULT_MAP_CENTER,
        opacity=DEFAULT_MAP_OPACITY,
        labels={"predicted_fare": "Predicted fare ($)"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=CHOROPLETH_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MAP_FONT, color=MAP_MUTED_COLOR, size=11),
        hoverlabel=dict(
            bgcolor="#141216",
            bordercolor="#332F36",
            font=dict(family=MAP_FONT, size=12, color=MAP_TEXT_COLOR),
        ),
        # The colour bar is chart furniture: legible, and then out of the way.
        coloraxis_colorbar=dict(
            title=dict(text="FARE ($)", font=dict(size=10, color=MAP_MUTED_COLOR)),
            tickfont=dict(size=10, color=MAP_MUTED_COLOR),
            outlinewidth=0,
            thickness=12,
            len=0.7,
            ticks="outside",
            ticklen=4,
            tickcolor="#332F36",
        ),
    )
    # The zone outlines default to white, which fights the dark basemap and makes 263
    # polygons read as a grid of boxes rather than a city.
    fig.update_traces(marker_line_color="#332F36", marker_line_width=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Predicted fare from {_zone_label(pu_location_id)} to each of the "
        f"{len(grid)} zones with known geometry, at the same time and passenger count. "
        "Distance is the straight line between centroids, so the absolute values run low."
    )
