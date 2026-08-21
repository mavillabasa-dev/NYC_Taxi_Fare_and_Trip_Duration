# dev_fixture_model.py — Generates a usable model artifact for T-110 / T-112 development.
#
# The bundle must be self-contained and loadable from the API without importing `src/`.
# It serializes a real `SelfContainedTaxiModel` with scikit-learn estimators,
# using cloudpickle to keep classes importable under `api.app.model.predictor`.

from __future__ import annotations

from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from api.app.model.predictor import SelfContainedTaxiModel

FEATURE_ORDER = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def _build_stub_model() -> SelfContainedTaxiModel:
    """Creates a minimal but valid bundle, equivalent to a production artifact."""
    rows = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, 265, size=rows),
            "DOLocationID": rng.integers(1, 265, size=rows),
            "tpep_pickup_datetime": pd.date_range("2022-05-01", periods=rows, freq="30min"),
            "passenger_count": rng.integers(1, 5, size=rows),
            "RatecodeID": rng.integers(1, 6, size=rows),
            "trip_distance": rng.uniform(0.5, 25.0, size=rows),
            "VendorID": rng.integers(1, 3, size=rows),
        }
    )

    centroid_lookup = {loc: (40.7 + (loc % 20) * 0.02, -74.0 + (loc % 15) * 0.03) for loc in range(1, 266)}
    target_encodings = {
        "PULocationID": {loc: 15.0 + loc % 12 for loc in range(1, 266)},
        "DOLocationID": {loc: 16.0 + loc % 10 for loc in range(1, 266)},
        "RatecodeID": {code: 14.0 + code for code in range(1, 7)},
    }

    model = SelfContainedTaxiModel(
        fare_model=LinearRegression(),
        duration_model=LinearRegression(),
        centroid_lookup=centroid_lookup,
        target_encodings=target_encodings,
        global_fare_mean=15.0,
        feature_names=None,
        version="1.0.0-lightgbm",
    )

    feat_df = model.transform_features(df)
    fare_target = 5.0 + 2.5 * feat_df["trip_distance"] + 3.0 * feat_df["passenger_count"]
    duration_target = 10.0 + 4.2 * feat_df["trip_distance"] + 2.0 * feat_df["is_weekend"]

    model.fare_model.fit(feat_df, fare_target)
    model.duration_model.fit(feat_df, duration_target)
    return model


def main() -> None:
    bundle = {
        "model": _build_stub_model(),
        "feature_order": FEATURE_ORDER,
        "version": "1.0.0-lightgbm",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        cloudpickle.dump(bundle, f)
    print(f"Production model written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
