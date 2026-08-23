"""scripts/dev_fixture_model.py — Generates a usable model artifact for T-110 / T-112 development.

Produces the same bundle shape as src/model_selection.export_production_model, but with
cheap LinearRegression estimators and synthetic centroids, so the API and the test suite
can run without the dataset and without a full training pass.

The class is imported as `app.model.predictor`, not `api.app.model.predictor`. Pickle
records the module path of the class, and inside the image `COPY . .` from ./api flattens
the tree so only `app.model.predictor` exists — an artifact pickled under `api.…` raises
ModuleNotFoundError in the container and fails tests/verify_isolation.py.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "api"

# api/ only — putting REPO_ROOT on the path makes `api.app.model.predictor` importable,
# and whichever spelling wins is the one baked into the pickle.
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from app.model.predictor import SelfContainedTaxiModel  # noqa: E402

FEATURE_ORDER = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
]

DEFAULT_OUTPUT_PATH = REPO_ROOT / "models" / "model.pkl"

# The Taxi Zone Shapefile only carries geometry for LocationID 1-263; 264 and 265 reach
# the lookup as NaN. Mirror that here so the fixture exercises the same edge the real
# artifact has.
LAST_ZONE_WITH_GEOMETRY = 263
MAX_LOCATION_ID = 265


def _build_stub_model() -> SelfContainedTaxiModel:
    """Creates a minimal but valid bundle, equivalent in shape to a production artifact."""
    rows = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, LAST_ZONE_WITH_GEOMETRY, size=rows),
            "DOLocationID": rng.integers(1, LAST_ZONE_WITH_GEOMETRY, size=rows),
            "tpep_pickup_datetime": pd.date_range("2022-05-01", periods=rows, freq="30min"),
            "passenger_count": rng.integers(1, 5, size=rows),
            "RatecodeID": rng.integers(1, 6, size=rows),
            "trip_distance": rng.uniform(0.5, 25.0, size=rows),
            "VendorID": rng.integers(1, 3, size=rows),
        }
    )

    centroid_lookup: dict[int, tuple[float, float]] = {
        loc: (40.7 + (loc % 20) * 0.02, -74.0 + (loc % 15) * 0.03)
        for loc in range(1, LAST_ZONE_WITH_GEOMETRY + 1)
    }
    for loc in range(LAST_ZONE_WITH_GEOMETRY + 1, MAX_LOCATION_ID + 1):
        centroid_lookup[loc] = (float("nan"), float("nan"))

    target_encodings = {
        "PULocationID": {loc: 15.0 + loc % 12 for loc in range(1, MAX_LOCATION_ID + 1)},
        "DOLocationID": {loc: 16.0 + loc % 10 for loc in range(1, MAX_LOCATION_ID + 1)},
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the bundle (default: models/model.pkl).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing artifact. Without this the script refuses, so a "
        "real trained model is never clobbered by accident.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite the existing artifact at {output_path}.\n"
            f"This path is also where src/model_selection.py writes the real trained "
            f"model. Re-run with --force if you meant to replace it, or pass --output "
            f"to write somewhere else."
        )

    bundle = {
        "model": _build_stub_model(),
        "feature_order": FEATURE_ORDER,
        "version": "1.0.0-lightgbm",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Development fixture model written to {output_path}")


if __name__ == "__main__":
    main()
