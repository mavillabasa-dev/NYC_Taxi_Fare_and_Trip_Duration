"""Streamlit application entry point (T-111 dashboard)."""

import streamlit as st

import api_client
from components import prediction_form

st.set_page_config(page_title="NYC Taxi Fare & Duration", page_icon="🚕")
st.title("NYC Taxi Fare & Trip Duration Predictor")

health = api_client.get_health()

if health.get("status") == "unreachable":
	st.error(f"Could not connect to API: {health.get('detail')}")
elif health.get("model_loaded"):
	st.success(f"API Connected — Model loaded (version {health.get('model_version')})")
else:
	st.warning(f"API Connected, but model is not loaded: {health.get('detail')}")

st.divider()
prediction_form.render()
