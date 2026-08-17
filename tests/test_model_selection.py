# tests/test_model_selection.py — Unit & Integration tests for T-109 Model Selection & Production Artifact
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app.model.predictor import SelfContainedTaxiModel
from api.app.model.services import REQUIRED_BUNDLE_KEYS, REQUEST_FEATURES
from src.config import MODELS_DIR, RANDOM_SEED
from src.model_selection import (
    compute_feature_importances,
    conduct_residual_analysis,
    verify_model_isolation,
)


@pytest.fixture
def sample_request_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PULocationID": 132,
                "DOLocationID": 236,
                "tpep_pickup_datetime": "2022-05-20T14:30:00",
                "passenger_count": 2,
                "RatecodeID": 1,
                "trip_distance": 6.2,
            },
            {
                "PULocationID": 161,
                "DOLocationID": 162,
                "tpep_pickup_datetime": "2022-05-21T08:15:00",
                "passenger_count": 1,
                "RatecodeID": 1,
                "trip_distance": 1.1,
            },
        ]
    )


def test_production_model_bundle_keys():
    """Verify that models/model.pkl contains all required bundle keys and request feature contracts."""
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        pytest.skip(f"Artifact {model_path} not found yet.")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    assert isinstance(bundle, dict)
    assert REQUIRED_BUNDLE_KEYS.issubset(bundle.keys())
    assert set(bundle["feature_order"]) == REQUEST_FEATURES
    assert "lightgbm" in bundle["version"].lower()
    assert hasattr(bundle["model"], "predict")


def test_self_contained_model_prediction_shape(sample_request_df):
    """Test that SelfContainedTaxiModel transforms raw features and returns 2D (N, 2) predictions."""
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        pytest.skip(f"Artifact {model_path} not found yet.")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    preds = model.predict(sample_request_df)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, 2)

    # Column 0: fare, Column 1: duration
    fare_preds = preds[:, 0]
    dur_preds = preds[:, 1]

    assert (fare_preds > 0).all()
    assert (dur_preds > 0).all()


def test_model_isolation_acceptance_test():
    """Acceptance test: Unpickling and predicting in a subprocess where src/ is absent from sys.path."""
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        pytest.skip(f"Artifact {model_path} not found yet.")

    passed = verify_model_isolation(model_path)
    assert passed is True


def test_real_model_api_serving_endpoint(sample_request_df):
    """Integration test: Test FastAPI app loaded with the real production model artifact."""
    import main

    with TestClient(main.app) as client:
        # 1. Health check
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_json = health_resp.json()
        assert health_json["status"] == "ok"
        assert health_json["model_loaded"] is True
        assert "lightgbm" in health_json["model_version"].lower()

        # 2. Prediction endpoint
        payload = sample_request_df.iloc[0].to_dict()
        pred_resp = client.post("/predict", json=payload)
        assert pred_resp.status_code == 200
        pred_json = pred_resp.json()

        assert "predicted_fare" in pred_json
        assert "predicted_duration_minutes" in pred_json
        assert pred_json["predicted_fare"] > 0
        assert pred_json["predicted_duration_minutes"] > 0
