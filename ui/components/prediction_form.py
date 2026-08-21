"""Prediction form: collects trip data and calls the API."""

import pandas as pd
import streamlit as st

import api_client
from components import choropleth, trip_map
from settings import (
	DEFAULT_DROPOFF_ZONE_ID,
	DEFAULT_PASSENGER_COUNT,
	DEFAULT_PICKUP_DATETIME,
	DEFAULT_PICKUP_ZONE_ID,
	DEFAULT_TRIP_DISTANCE_MILES,
	HTTP_STATUS_OK,
	HTTP_STATUS_UNAVAILABLE,
	HTTP_STATUS_UNPROCESSABLE,
	HTTP_STATUS_UNREACHABLE,
	PASSENGER_MAX,
	PASSENGER_MIN,
)
from zones import (
	RATECODE_LABELS,
	estimate_trip_distance,
	infer_ratecode,
	load_zones,
)


def render() -> None:
	zones = load_zones()
	labels = zones["label"].tolist()
	label_to_id = dict(zip(zones["label"], zones["LocationID"]))
	id_to_label = dict(zip(zones["LocationID"], zones["label"]))

	pickup_default_index = labels.index(id_to_label[DEFAULT_PICKUP_ZONE_ID])
	dropoff_default_index = labels.index(id_to_label[DEFAULT_DROPOFF_ZONE_ID])

	# Pickup/Dropoff (and inferred Rate code / Trip distance) remain
	# OUTSIDE st.form: widgets inside a form do not trigger a rerun until
	# submit, and these "auto" fields need to update live as soon as a zone
	# changes, not only upon clicking predict.
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		pu_label = st.selectbox("Pickup zone", labels, index=pickup_default_index)
	with col2:
		do_label = st.selectbox("Dropoff zone", labels, index=dropoff_default_index)

	pu_id = int(label_to_id[pu_label])
	do_id = int(label_to_id[do_label])
	inferred_ratecode = infer_ratecode(pu_id, do_id)
	inferred_ratecode_label = f"{inferred_ratecode} - {RATECODE_LABELS[inferred_ratecode]}"
	estimated_distance = estimate_trip_distance(pu_id, do_id)

	with col3:
		st.selectbox(
			"Rate code (auto)", options=[inferred_ratecode_label], index=0, disabled=True
		)
	with col4:
		st.number_input(
			"Trip distance (auto, miles)", value=estimated_distance, disabled=True
		)

	pu_row = zones[zones.LocationID == pu_id].iloc[0]
	do_row = zones[zones.LocationID == do_id].iloc[0]
	if (
		estimated_distance == DEFAULT_TRIP_DISTANCE_MILES
		and (pd.isna(pu_row.longitude) or pd.isna(do_row.longitude))
	):
		st.caption(
			"⚠️ One of the selected zones lacks known coordinates — a fallback "
			f"value ({DEFAULT_TRIP_DISTANCE_MILES} mi) was used instead of a real estimate."
		)

	with st.form("prediction_form"):
		col1, col2 = st.columns(2)
		with col1:
			pickup_dt = st.text_input("Pickup datetime (ISO)", DEFAULT_PICKUP_DATETIME)
		with col2:
			passengers = st.number_input(
				"Passengers",
				min_value=PASSENGER_MIN,
				max_value=PASSENGER_MAX,
				value=DEFAULT_PASSENGER_COUNT,
			)
		submitted = st.form_submit_button("Predict")

	if submitted:
		payload = {
			"PULocationID": pu_id,
			"DOLocationID": do_id,
			"tpep_pickup_datetime": pickup_dt,
			"passenger_count": passengers,
			"RatecodeID": inferred_ratecode,
			"trip_distance": estimated_distance,
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