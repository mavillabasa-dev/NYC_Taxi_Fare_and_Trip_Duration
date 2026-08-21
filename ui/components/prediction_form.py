"""Prediction form: collects trip data and calls the API."""

import streamlit as st

import api_client
from components import choropleth, trip_map
from settings import (
	DEFAULT_DROPOFF_ZONE_ID,
	DEFAULT_PASSENGER_COUNT,
	DEFAULT_PICKUP_DATETIME,
	DEFAULT_PICKUP_ZONE_ID,
	DEFAULT_RATECODE,
	DEFAULT_TRIP_DISTANCE_MILES,
	HTTP_STATUS_OK,
	HTTP_STATUS_UNAVAILABLE,
	HTTP_STATUS_UNPROCESSABLE,
	HTTP_STATUS_UNREACHABLE,
	PASSENGER_MAX,
	PASSENGER_MIN,
	RATECODES,
	TRIP_DISTANCE_MAX,
	TRIP_DISTANCE_MIN,
)
from zones import load_zones


def render() -> None:
	zones = load_zones()
	labels = zones["label"].tolist()
	label_to_id = dict(zip(zones["label"], zones["LocationID"]))
	id_to_label = dict(zip(zones["LocationID"], zones["label"]))

	pickup_default_index = labels.index(id_to_label[DEFAULT_PICKUP_ZONE_ID])
	dropoff_default_index = labels.index(id_to_label[DEFAULT_DROPOFF_ZONE_ID])

	with st.form("prediction_form"):
		col1, col2 = st.columns(2)
		with col1:
			pu_label = st.selectbox("Pickup zone", labels, index=pickup_default_index)
			pickup_dt = st.text_input("Pickup datetime (ISO)", DEFAULT_PICKUP_DATETIME)
			passengers = st.number_input(
				"Passengers",
				min_value=PASSENGER_MIN,
				max_value=PASSENGER_MAX,
				value=DEFAULT_PASSENGER_COUNT,
			)
		with col2:
			do_label = st.selectbox("Dropoff zone", labels, index=dropoff_default_index)
			ratecode = st.selectbox("Rate code", RATECODES, index=RATECODES.index(DEFAULT_RATECODE))
			distance = st.number_input(
				"Trip distance (miles)",
				min_value=TRIP_DISTANCE_MIN,
				max_value=TRIP_DISTANCE_MAX,
				value=DEFAULT_TRIP_DISTANCE_MILES,
			)
		submitted = st.form_submit_button("Predict")

	if submitted:
		payload = {
			"PULocationID": int(label_to_id[pu_label]),
			"DOLocationID": int(label_to_id[do_label]),
			"tpep_pickup_datetime": pickup_dt,
			"passenger_count": passengers,
			"RatecodeID": ratecode,
			"trip_distance": distance,
		}
		status_code, body = api_client.predict(payload)
		# Saved in session_state because the result must survive reruns
		# triggered by OTHER widgets (e.g. the choropleth checkbox) —
		# `submitted` is only True on the exact rerun of clicking "Predict".
		st.session_state["last_prediction"] = {
			"status_code": status_code,
			"body": body,
			"payload": payload,
		}

	if "last_prediction" in st.session_state:
		result = st.session_state["last_prediction"]
		_render_result(result["status_code"], result["body"], result["payload"])


def _render_result(status_code: int, body: dict, payload: dict) -> None:
	if status_code == HTTP_STATUS_OK:
		col1, col2 = st.columns(2)
		col1.metric("Predicted fare", f"${body['predicted_fare']:.2f}")
		col2.metric("Predicted duration", f"{body['predicted_duration_minutes']:.1f} min")
		trip_map.render(payload["PULocationID"], payload["DOLocationID"])

		if st.checkbox("Show predicted fares to all NYC dropoff zones"):
			choropleth.render(
				payload["PULocationID"],
				payload["tpep_pickup_datetime"],
				payload["passenger_count"],
				payload["RatecodeID"],
			)
	elif status_code == HTTP_STATUS_UNREACHABLE:
		st.error(f"Could not connect to API. Is the server running? Details: {body.get('detail')}")
	elif status_code == HTTP_STATUS_UNAVAILABLE:
		st.error(f"Model is not loaded yet in API. Details: {body.get('detail')}")
	elif status_code == HTTP_STATUS_UNPROCESSABLE:
		st.error(f"Invalid input:\n\n{_format_validation_errors(body.get('detail'))}")
	else:
		st.error(f"Unexpected error ({status_code}): {body.get('detail')}")


def _format_validation_errors(detail) -> str:
	if not isinstance(detail, list):
		return str(detail)

	lines = []
	for error in detail:
		field = error.get("loc", ["?"])[-1]
		message = error.get("msg", "invalid value")
		lines.append(f"- **{field}**: {message}")
	return "\n".join(lines)