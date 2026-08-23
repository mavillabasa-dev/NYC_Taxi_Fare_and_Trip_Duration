"""Streamlit application entry point (T-111 dashboard)."""

import api_client
import streamlit as st
import theme
from components import prediction_form

st.set_page_config(
    page_title="NYC Taxi Fare & Duration",
    layout="centered",
)

# Must come immediately after set_page_config: Streamlit renders the page shell on the
# first call, and CSS injected later repaints visibly.
theme.apply()

theme.checker()
theme.eyebrow("Fare and trip duration · New York City")
st.title("What will this ride cost?")
st.markdown(
    '<p style="color:#A9A294;font-size:1.02rem;margin:-.2rem 0 1.4rem;max-width:52ch">'
    "Predicted before the meter starts, from the pickup zone, the destination and the "
    "time of day."
    "</p>",
    unsafe_allow_html=True,
)

health = api_client.get_health()

# The API being down is the expected failure in a live demo, so it gets a plain sentence
# rather than a stack trace. The healthy case is deliberately quiet — a green box on
# every load trains people to stop reading the status line.
if health.get("status") == "unreachable":
    st.error(
        f"Cannot reach the prediction service. Is it running on {api_client.settings.api_url}?"
        f"\n\n`{health.get('detail')}`"
    )
elif not health.get("model_loaded"):
    st.warning(f"The service is up but no model is loaded: {health.get('detail')}")
else:
    theme.eyebrow(f"model {health.get('model_version', 'unknown')} · ready")

theme.checker(small=True)
prediction_form.render()
