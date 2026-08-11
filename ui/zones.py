"""Carga y cachea el catálogo de zonas de NYC (nombre, borough, coordenadas)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from settings import settings


@st.cache_data
def load_zones() -> pd.DataFrame:
	centroids_path = Path(settings.dataset_dir) / "taxi_zone_centroids.csv"
	df = pd.read_csv(centroids_path)
	df["Borough"] = df["Borough"].fillna("Unknown")
	df["Zone"] = df["Zone"].fillna("Unknown")
	df["label"] = df["Zone"] + " (" + df["Borough"] + ")"
	return df.sort_values("label").reset_index(drop=True)
