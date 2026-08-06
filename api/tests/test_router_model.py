"""Pruebas del contrato HTTP y del artefacto provisional de T-110."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class FixtureModel:
	def predict(self, features):
		assert list(features.columns) == [
			"PULocationID",
			"DOLocationID",
			"tpep_pickup_datetime",
			"passenger_count",
			"RatecodeID",
			"trip_distance",
		]
		return np.array([[18.75, 27.5]])


@pytest.fixture
def valid_payload() -> dict:
	return {
		"PULocationID": 132,
		"DOLocationID": 236,
		"tpep_pickup_datetime": "2022-05-20T14:30:00",
		"passenger_count": 2,
		"RatecodeID": 1,
		"trip_distance": 6.2,
	}


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
	artifact_path = tmp_path / "model.pkl"
	bundle = {
		"model": FixtureModel(),
		"feature_order": [
			"PULocationID",
			"DOLocationID",
			"tpep_pickup_datetime",
			"passenger_count",
			"RatecodeID",
			"trip_distance",
		],
		"version": "fixture-1.0",
	}
	with artifact_path.open("wb") as artifact_file:
		pickle.dump(bundle, artifact_file)

	monkeypatch.setattr(
		main, "settings", SimpleNamespace(model_path=str(artifact_path))
	)
	with TestClient(main.app) as test_client:
		yield test_client


def test_health_reports_loaded_model(client: TestClient):
	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {
		"status": "ok",
		"model_loaded": True,
		"model_version": "fixture-1.0",
		"detail": None,
	}


def test_predict_returns_both_targets(client: TestClient, valid_payload: dict):
	response = client.post("/predict", json=valid_payload)

	assert response.status_code == 200
	assert response.json() == {
		"predicted_fare": 18.75,
		"predicted_duration_minutes": 27.5,
		"model_version": "fixture-1.0",
	}


@pytest.mark.parametrize(
	("field", "value"),
	[
		("PULocationID", 0),
		("DOLocationID", 266),
		("passenger_count", 0),
		("RatecodeID", 99),
		("trip_distance", 0),
		("tpep_pickup_datetime", "not-a-date"),
	],
)
def test_invalid_payload_returns_422(
	client: TestClient, valid_payload: dict, field: str, value
):
	valid_payload[field] = value
	response = client.post("/predict", json=valid_payload)

	assert response.status_code == 422
	assert response.json()["detail"]


def test_docs_are_available(client: TestClient):
	assert client.get("/docs").status_code == 200


def test_missing_model_returns_degraded_health_and_503(tmp_path, monkeypatch):
	missing_path = tmp_path / "missing.pkl"
	monkeypatch.setattr(main, "settings", SimpleNamespace(model_path=str(missing_path)))

	with TestClient(main.app) as test_client:
		health_response = test_client.get("/health")
		predict_response = test_client.post(
			"/predict",
			json={
				"PULocationID": 132,
				"DOLocationID": 236,
				"tpep_pickup_datetime": "2022-05-20T14:30:00",
				"passenger_count": 2,
				"RatecodeID": 1,
				"trip_distance": 6.2,
			},
		)

	assert health_response.status_code == 200
	assert health_response.json()["model_loaded"] is False
	assert health_response.json()["status"] == "degraded"
	assert predict_response.status_code == 503
