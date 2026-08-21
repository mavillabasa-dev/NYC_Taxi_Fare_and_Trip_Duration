"""Model artifact loading and prediction service."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from app.model.schema import PredictionRequest, PredictionResponse

REQUIRED_BUNDLE_KEYS = {"model", "feature_order", "version"}
REQUEST_FEATURES = {
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
}


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_order: list[str] = []
        self.version: str | None = None
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_path: str) -> None:
        self.model = None
        self.feature_order = []
        self.version = None
        self.load_error = None

        path = Path(model_path)
        try:
            with path.open("rb") as artifact_file:
                bundle = pickle.load(artifact_file)
            if not isinstance(bundle, dict):
                raise ValueError("Model artifact must be a dictionary bundle")

            missing_keys = REQUIRED_BUNDLE_KEYS - bundle.keys()
            if missing_keys:
                raise ValueError(
                    f"Model artifact is missing keys: {sorted(missing_keys)}"
                )

            feature_order = list(bundle["feature_order"])
            if set(feature_order) != REQUEST_FEATURES:
                raise ValueError(
                    "Model feature_order must contain exactly the API request features"
                )

            self.model = bundle["model"]
            self.feature_order = feature_order
            self.version = str(bundle["version"])

            # Pre-warm booster & validation buffers
            if hasattr(self.model, "warm_up"):
                try:
                    self.model.warm_up()
                except Exception:
                    pass
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        if not self.is_loaded or self.version is None:
            raise RuntimeError(self.load_error or "Model is not loaded")

        input_data = payload.model_dump()

        # High-performance fast-path inference when supported by the model
        if callable(getattr(self.model, "predict_fast", None)):
            try:
                predicted_fare, predicted_duration = self.model.predict_fast(input_data)
                return PredictionResponse(
                    predicted_fare=float(predicted_fare),
                    predicted_duration_minutes=float(predicted_duration),
                    model_version=self.version,
                )
            except Exception:
                # Graceful fallback to DataFrame prediction path
                pass

        features = pd.DataFrame(
            [[input_data[name] for name in self.feature_order]],
            columns=self.feature_order,
        )
        prediction = self.model.predict(features)

        try:
            predicted_fare, predicted_duration = prediction[0]
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
				"Model must return [fare, duration] for each input row"
            ) from exc

        return PredictionResponse(
            predicted_fare=float(predicted_fare),
            predicted_duration_minutes=float(predicted_duration),
            model_version=self.version,
        )


model_service = ModelService()
