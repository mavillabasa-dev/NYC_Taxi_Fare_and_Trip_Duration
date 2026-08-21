"""Carga y cachea el catálogo de zonas de NYC (nombre, borough, coordenadas)."""

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
