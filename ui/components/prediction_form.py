"""Formulario de predicción: recolecta los datos del viaje y llama a la API."""

import streamlit as st

import api_client
from components import choropleth, trip_map
from zones import load_zones

PASSENGER_MIN, PASSENGER_MAX = 1, 9
RATECODES = [1, 2, 3, 4, 5, 6]
TRIP_DISTANCE_MIN, TRIP_DISTANCE_MAX = 0.1, 150.0

DEFAULT_PICKUP_ZONE_ID = 132  # JFK Airport (Queens)
DEFAULT_DROPOFF_ZONE_ID = 236  # Upper East Side North (Manhattan)


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
			pickup_dt = st.text_input("Pickup datetime (ISO)", "2022-05-20T14:30:00")
			passengers = st.number_input(
				"Passengers", min_value=PASSENGER_MIN, max_value=PASSENGER_MAX, value=1
			)
		with col2:
			do_label = st.selectbox("Dropoff zone", labels, index=dropoff_default_index)
			ratecode = st.selectbox("Rate code", RATECODES, index=0)
			distance = st.number_input(
				"Trip distance (miles)",
				min_value=TRIP_DISTANCE_MIN,
				max_value=TRIP_DISTANCE_MAX,
				value=3.0,
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
		# Se guarda en session_state porque el resultado tiene que sobrevivir a
		# reruns disparados por OTROS widgets (ej. el checkbox del choropleth) —
		# `submitted` solo es True en el rerun exacto del click en "Predict".
		st.session_state["last_prediction"] = {
			"status_code": status_code,
			"body": body,
			"payload": payload,
		}

	if "last_prediction" in st.session_state:
		result = st.session_state["last_prediction"]
		_render_result(result["status_code"], result["body"], result["payload"])


def _render_result(status_code: int, body: dict, payload: dict) -> None:
	if status_code == 200:
		col1, col2 = st.columns(2)
		col1.metric("Predicted fare", f"${body['predicted_fare']:.2f}")
		col2.metric("Predicted duration", f"{body['predicted_duration_minutes']:.1f} min")
		trip_map.render(payload["PULocationID"], payload["DOLocationID"])

		if st.checkbox("Mostrar choropleth de tarifas por zona (llama a la API ~263 veces)"):
			choropleth.render(
				payload["PULocationID"],
				payload["tpep_pickup_datetime"],
				payload["passenger_count"],
				payload["RatecodeID"],
			)
	elif status_code == 0:
		st.error(f"No se pudo contactar la API. ¿Está corriendo? Detalle: {body.get('detail')}")
	elif status_code == 503:
		st.error(f"El modelo todavía no está cargado en la API. Detalle: {body.get('detail')}")
	elif status_code == 422:
		st.error(f"Datos inválidos:\n\n{_format_validation_errors(body.get('detail'))}")
	else:
		st.error(f"Error inesperado ({status_code}): {body.get('detail')}")


def _format_validation_errors(detail) -> str:
	if not isinstance(detail, list):
		return str(detail)

	lines = []
	for error in detail:
		field = error.get("loc", ["?"])[-1]
		message = error.get("msg", "invalid value")
		lines.append(f"- **{field}**: {message}")
	return "\n".join(lines)