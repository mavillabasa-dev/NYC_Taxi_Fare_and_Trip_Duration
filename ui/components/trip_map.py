"""Point map: displays pickup and dropoff of the predicted trip on NYC."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from settings import (
    DEFAULT_POINT_MAP_ZOOM,
    DROPOFF_MARKER_COLOR,
    MAP_FONT,
    MAP_STYLE,
    MAP_TEXT_COLOR,
    MARKER_SIZE,
    PICKUP_MARKER_COLOR,
    POINT_MAP_HEIGHT,
    TRIP_LINE_COLOR,
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

    fig = go.Figure()

    # The straight line between the two centroids, drawn faintly. It is not a route and
    # must not be mistaken for one - but it is not decoration either: this exact segment
    # is the haversine distance the model is being fed, so drawing it shows what the
    # prediction is actually based on.
    fig.add_trace(
        go.Scattermapbox(
            lat=[pu_row.latitude, do_row.latitude],
            lon=[pu_row.longitude, do_row.longitude],
            mode="lines",
            line=dict(width=2, color=TRIP_LINE_COLOR),
            opacity=0.45,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=[pu_row.latitude, do_row.latitude],
            lon=[pu_row.longitude, do_row.longitude],
            mode="markers+text",
            text=["PICKUP", "DROPOFF"],
            textposition="top center",
            textfont=dict(family=MAP_FONT, size=11, color=MAP_TEXT_COLOR),
            marker=dict(size=MARKER_SIZE, color=[PICKUP_MARKER_COLOR, DROPOFF_MARKER_COLOR]),
            # Without this, Plotly displays raw lat/lon in hover by default.
            # hoverinfo="text" instructs it to use hovertext instead.
            hovertext=[f"Pickup: {pu_row.label}", f"Dropoff: {do_row.label}"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        mapbox_style=MAP_STYLE,
        mapbox_zoom=DEFAULT_POINT_MAP_ZOOM,
        mapbox_center={
            "lat": (pu_row.latitude + do_row.latitude) / 2,
            "lon": (pu_row.longitude + do_row.longitude) / 2,
        },
        margin=dict(l=0, r=0, t=0, b=0),
        height=POINT_MAP_HEIGHT,
        # Transparent so the map sits on the page surface rather than on a white slab.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#141216",
            bordercolor="#332F36",
            font=dict(family=MAP_FONT, size=12, color=MAP_TEXT_COLOR),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The line is the straight-line distance between zone centroids, which is the "
        "distance the model receives. A real route would be longer."
    )
