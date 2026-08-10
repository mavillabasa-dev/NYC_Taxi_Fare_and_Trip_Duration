"""Cliente HTTP para consumir la API de predicción (T-110)."""

import requests

from settings import settings

TIMEOUT_SECONDS = 5


def get_health() -> dict:
	try:
		response = requests.get(f"{settings.api_url}/health", timeout=TIMEOUT_SECONDS)
		return response.json()
	except requests.exceptions.RequestException as exc:
		return {
			"status": "unreachable",
			"model_loaded": False,
			"model_version": None,
			"detail": str(exc),
		}


def predict(payload: dict) -> tuple[int, dict]:
	try:
		response = requests.post(
			f"{settings.api_url}/predict", json=payload, timeout=TIMEOUT_SECONDS
		)
	except requests.exceptions.RequestException as exc:
		return 0, {"detail": str(exc)}

	try:
		body = response.json()
	except ValueError:
		body = {"detail": response.text}
	return response.status_code, body
