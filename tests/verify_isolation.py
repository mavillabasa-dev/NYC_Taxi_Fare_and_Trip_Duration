"""tests/verify_isolation.py — Standalone acceptance verification for production model artifact.

This script executes in an isolated environment where `src/` is absent from `sys.path`.
It verifies that `models/model.pkl` can be unpickled and executed using only the container runtime
dependencies (located in `api/`).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def verify_artifact_isolation(model_path: str, api_dir: str) -> None:
    """Verifies that model_path unpickles and predicts without any dependency on src/."""
    # 1. Strip repository root and src/ paths from sys.path
    repo_root = str(Path(api_dir).resolve().parent)
    sys.path = [
        p
        for p in sys.path
        if p not in ("", ".")
        and not p.endswith("NYC_Taxi_Fare_and_Trip_Duration")
        and "/src" not in p
        and p != repo_root
    ]
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)

    # 2. Strict assertion: importing src MUST fail in isolation
    try:
        import src  # type: ignore # noqa: F401

        raise RuntimeError("ISOLATION FAILURE: src package was successfully imported!")
    except ImportError:
        pass

    # 3. Load model artifact dictionary
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict):
        raise TypeError(f"Artifact must be a dictionary, got {type(bundle)}")

    for key in ("model", "feature_order", "version"):
        if key not in bundle:
            raise KeyError(f"Missing required key '{key}' in artifact bundle")

    # 4. Construct raw sample request DataFrame
    feature_order = bundle["feature_order"]
    sample_df = pd.DataFrame(
        [
            {
                "PULocationID": 132,
                "DOLocationID": 236,
                "tpep_pickup_datetime": "2022-05-20T14:30:00",
                "passenger_count": 2,
                "RatecodeID": 1,
                "trip_distance": 6.2,
            }
        ]
    )[feature_order]

    # 5. Execute prediction
    model = bundle["model"]
    preds = model.predict(sample_df)

    if not isinstance(preds, np.ndarray) or preds.shape != (1, 2):
        raise ValueError(f"Expected prediction shape (1, 2), got {getattr(preds, 'shape', None)}")

    fare, duration = float(preds[0, 0]), float(preds[0, 1])
    if fare <= 0 or duration <= 0:
        raise ValueError(f"Invalid non-positive predictions: fare={fare}, duration={duration}")

    print(
        f"ISOLATION_TEST_PASSED: fare=${fare:.2f}, duration={duration:.2f} mins, version={bundle['version']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify model artifact isolation")
    parser.add_argument("--model-path", default="models/model.pkl", help="Path to model.pkl")
    parser.add_argument("--api-dir", default="api", help="Path to api directory")
    args = parser.parse_args()

    model_abs = os.path.abspath(args.model_path)
    api_abs = os.path.abspath(args.api_dir)

    verify_artifact_isolation(model_abs, api_abs)


if __name__ == "__main__":
    main()
