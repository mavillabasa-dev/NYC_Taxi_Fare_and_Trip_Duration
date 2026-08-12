"""Punto de entrada de la aplicación Streamlit (dashboard de T-111)."""

import streamlit as st

import api_client
from components import prediction_form

st.set_page_config(page_title="NYC Taxi Fare & Duration", page_icon="🚕")
st.title("NYC Taxi Fare & Trip Duration Predictor")

health = api_client.get_health()

if health.get("status") == "unreachable":
	st.error(f"No se pudo contactar la API: {health.get('detail')}")
elif health.get("model_loaded"):
	st.success(f"API conectada — modelo cargado (versión {health.get('model_version')})")
else:
	st.warning(f"API conectada, pero el modelo no está cargado: {health.get('detail')}")

st.divider()
prediction_form.render()
