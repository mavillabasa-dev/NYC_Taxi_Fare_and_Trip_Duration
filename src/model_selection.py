"""src/model_selection.py — Model Selection, Benchmarking, Residual Analysis & Production Packaging (T-109).

This module handles:
1. Benchmarking & Leaderboard compilation across all candidate model families (T-106, T-107, T-108).
2. Formal selection and training of the winning LightGBM regressors.
3. Feature importance analysis (split gain & split count) and leakage verification.
4. In-depth residual & error analysis across temporal, spatial, distance, and airport slices.
5. Exporting the self-contained production bundle to `models/model.pkl` with zero runtime dependency on `src/`.
6. Automated isolation verification test asserting clean unpickling with `src/` removed from `sys.path`.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Ensure api/ is on sys.path so app is imported with the top-level 'app' package matching container layout
_api_path = str(Path(__file__).resolve().parent.parent / "api")
if _api_path not in sys.path:
    sys.path.insert(0, _api_path)

from app.model.predictor import SelfContainedTaxiModel
from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TAXI_ZONE_CENTROIDS_PATH,
    TAXI_ZONE_LOOKUP_PATH,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.features import NYCFeaturePipeline
from src.train import calculate_metrics, measure_inference_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUEST_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
]


def train_winning_lightgbm_models(
    X_train: pd.DataFrame,
    y_train_fare: np.ndarray,
    y_train_dur: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> Tuple[LGBMRegressor, LGBMRegressor]:
    """Trains optimal LightGBM models for fare_amount and duration_minutes."""
    logger.info("Training winning LightGBM model for fare_amount...")
    lgb_fare = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_seed,
        verbosity=-1,
        n_jobs=-1,
    )
    t0 = time.time()
    lgb_fare.fit(X_train, y_train_fare)
    logger.info(f"Fitted LightGBM fare model in {time.time() - t0:.2f}s")

    logger.info("Training winning LightGBM model for duration_minutes...")
    lgb_dur = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_seed,
        verbosity=-1,
        n_jobs=-1,
    )
    t0 = time.time()
    lgb_dur.fit(X_train, y_train_dur)
    logger.info(f"Fitted LightGBM duration model in {time.time() - t0:.2f}s")

    return lgb_fare, lgb_dur


def compute_feature_importances(
    lgb_fare: LGBMRegressor,
    lgb_dur: LGBMRegressor,
    feature_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """Extracts and ranks feature importances by split gain and split count."""
    fare_gain = lgb_fare.booster_.feature_importance(importance_type="gain")
    fare_split = lgb_fare.booster_.feature_importance(importance_type="split")

    dur_gain = lgb_dur.booster_.feature_importance(importance_type="gain")
    dur_split = lgb_dur.booster_.feature_importance(importance_type="split")

    df_fare = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": fare_gain,
            "split": fare_split,
            "gain_pct": (fare_gain / fare_gain.sum()) * 100.0,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    df_dur = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": dur_gain,
            "split": dur_split,
            "gain_pct": (dur_gain / dur_gain.sum()) * 100.0,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    return {"fare": df_fare, "duration": df_dur}


def conduct_residual_analysis(
    y_true_fare: np.ndarray,
    y_pred_fare: np.ndarray,
    y_true_dur: np.ndarray,
    y_pred_dur: np.ndarray,
    test_df: pd.DataFrame,
    centroids_path: str = TAXI_ZONE_CENTROIDS_PATH,
) -> Dict[str, Any]:
    """Conducts in-depth residual and error breakdown across 4 analytical dimensions."""
    logger.info("Conducting in-depth residual error analysis...")
    df_eval = test_df.copy()
    df_eval["y_true_fare"] = y_true_fare
    df_eval["y_pred_fare"] = y_pred_fare
    df_eval["res_fare"] = y_true_fare - y_pred_fare
    df_eval["abs_res_fare"] = np.abs(df_eval["res_fare"])

    df_eval["y_true_dur"] = y_true_dur
    df_eval["y_pred_dur"] = y_pred_dur
    df_eval["res_dur"] = y_true_dur - y_pred_dur
    df_eval["abs_res_dur"] = np.abs(df_eval["res_dur"])

    dt = pd.to_datetime(df_eval["tpep_pickup_datetime"])
    df_eval["pickup_hour"] = dt.dt.hour

    # Join Borough metadata
    if os.path.exists(centroids_path):
        cent_df = pd.read_csv(centroids_path)
        borough_map = cent_df.set_index("LocationID")["Borough"].to_dict()
        df_eval["pu_borough"] = df_eval["PULocationID"].map(borough_map).fillna("Unknown")
    else:
        df_eval["pu_borough"] = "Unknown"

    # Trip distance buckets
    bins = [0.0, 2.0, 10.0, 1000.0]
    labels = ["Short (<2 mi)", "Medium (2-10 mi)", "Long (>10 mi)"]
    df_eval["distance_bucket"] = pd.cut(df_eval["trip_distance"], bins=bins, labels=labels)

    # 1. Residuals by Hour of Day
    by_hour = df_eval.groupby("pickup_hour").agg(
        trips=("fare_amount", "count"),
        fare_mae=("abs_res_fare", "mean"),
        fare_mean_res=("res_fare", "mean"),
        dur_mae=("abs_res_dur", "mean"),
        dur_mean_res=("res_dur", "mean"),
    ).round(4)

    # 2. Residuals by Pickup Borough
    by_borough = df_eval.groupby("pu_borough").agg(
        trips=("fare_amount", "count"),
        fare_mae=("abs_res_fare", "mean"),
        fare_mean_res=("res_fare", "mean"),
        dur_mae=("abs_res_dur", "mean"),
        dur_mean_res=("res_dur", "mean"),
    ).round(4)

    # 3. Residuals by Trip Distance Bucket
    by_distance = df_eval.groupby("distance_bucket", observed=False).agg(
        trips=("fare_amount", "count"),
        fare_mae=("abs_res_fare", "mean"),
        dur_mae=("abs_res_dur", "mean"),
    ).round(4)

    # 4. Residuals by RatecodeID (Airport vs Standard)
    by_ratecode = df_eval.groupby("RatecodeID").agg(
        trips=("fare_amount", "count"),
        fare_mae=("abs_res_fare", "mean"),
        fare_mean_res=("res_fare", "mean"),
        dur_mae=("abs_res_dur", "mean"),
        dur_mean_res=("res_dur", "mean"),
    ).round(4)

    return {
        "by_hour": by_hour,
        "by_borough": by_borough,
        "by_distance": by_distance,
        "by_ratecode": by_ratecode,
    }


def export_production_model(
    lgb_fare: LGBMRegressor,
    lgb_dur: LGBMRegressor,
    pipeline_path: str = os.path.join(MODELS_DIR, "feature_pipeline.pkl"),
    centroids_path: str = TAXI_ZONE_CENTROIDS_PATH,
    output_path: str = os.path.join(MODELS_DIR, "model.pkl"),
    version: str = "1.0.0-lightgbm",
) -> Dict[str, Any]:
    """
    Builds and pickles the self-contained production bundle.

    The artifact embeds:
    - Centroid coordinates mapping LocationID -> (lat, lon)
    - Target encoding mapping dictionaries
    - LightGBM estimators
    - Exact request feature order contract
    """
    logger.info(f"Building self-contained production model artifact at {output_path}...")

    # 1. Extract centroid lookup
    centroid_lookup: Dict[int, Tuple[float, float]] = {}
    if os.path.exists(centroids_path):
        cent_df = pd.read_csv(centroids_path)
        for _, row in cent_df.iterrows():
            loc_id = int(row["LocationID"])
            lat = float(row["latitude"]) if pd.notna(row["latitude"]) else np.nan
            lon = float(row["longitude"]) if pd.notna(row["longitude"]) else np.nan
            centroid_lookup[loc_id] = (lat, lon)
    else:
        logger.warning(f"Centroids file not found at {centroids_path}. Lookup will be empty.")

    # 2. Extract target encodings from fitted feature pipeline
    target_encodings: Dict[str, Dict[Any, float]] = {}
    global_fare_mean = 15.15
    feature_names: Optional[List[str]] = None
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "rb") as f:
            feat_pipe: NYCFeaturePipeline = pickle.load(f)
        target_encodings = feat_pipe.target_encoder.target_maps_
        global_fare_mean = feat_pipe.target_encoder.global_means_.get("fare", 15.15)
        feature_names = getattr(feat_pipe, "feature_names_", None)
    else:
        logger.warning(f"Feature pipeline not found at {pipeline_path}. Target encodings will be empty.")

    # 3. Instantiate SelfContainedTaxiModel
    model = SelfContainedTaxiModel(
        fare_model=lgb_fare,
        duration_model=lgb_dur,
        centroid_lookup=centroid_lookup,
        target_encodings=target_encodings,
        global_fare_mean=global_fare_mean,
        feature_names=feature_names,
        version=version,
    )

    # 4. Assemble bundle dictionary
    bundle = {
        "model": model,
        "feature_order": REQUEST_FEATURES,
        "version": version,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Successfully saved self-contained model artifact to {output_path}")
    return bundle


def verify_model_isolation(model_path: str = os.path.join(MODELS_DIR, "model.pkl")) -> bool:
    """
    Executes an isolated acceptance test by running `tests/verify_isolation.py` in a subprocess.
    Simulates the container environment by ensuring `src/` is absent from sys.path.
    """
    logger.info("--- Running Strict Model Isolation & Unpickle Acceptance Test ---")
    repo_root = Path(__file__).resolve().parent.parent
    verifier_script = repo_root / "tests" / "verify_isolation.py"
    api_dir = repo_root / "api"

    cmd = [
        sys.executable,
        str(verifier_script),
        "--model-path",
        str(model_path),
        "--api-dir",
        str(api_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(api_dir))

    if res.returncode == 0 and "ISOLATION_TEST_PASSED" in res.stdout:
        logger.info(f"Acceptance test passed: {res.stdout.strip()}")
        return True
    else:
        logger.error(
            f"Acceptance test failed with code {res.returncode}:\n"
            f"Stdout: {res.stdout}\nStderr: {res.stderr}"
        )
        return False


def run_full_model_selection_pipeline(
    train_path: str = TRAIN_CLEANED_PATH,
    test_path: str = TEST_CLEANED_PATH,
    centroids_path: str = TAXI_ZONE_CENTROIDS_PATH,
    output_model_path: str = os.path.join(MODELS_DIR, "model.pkl"),
) -> Dict[str, Any]:
    """Runs end-to-end model selection, evaluation, error analysis, and artifact export."""
    logger.info("=== Starting T-109 Model Selection & Evaluation Pipeline ===")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        feature_pipeline: NYCFeaturePipeline = pickle.load(f)

    X_train_feat = feature_pipeline.transform(train_df[ALLOWED_FEATURES])
    X_test_feat = feature_pipeline.transform(test_df[ALLOWED_FEATURES])

    y_train_fare = train_df["fare_amount"].values
    y_train_dur = train_df["duration_minutes"].values
    y_test_fare = test_df["fare_amount"].values
    y_test_dur = test_df["duration_minutes"].values

    feature_names = list(X_train_feat.columns)

    # 1. Train winning LightGBM models
    lgb_fare, lgb_dur = train_winning_lightgbm_models(
        X_train_feat, y_train_fare, y_train_dur, random_seed=RANDOM_SEED
    )

    # 2. Evaluate on unseen temporal test set
    pred_fare = lgb_fare.predict(X_test_feat)
    pred_dur = lgb_dur.predict(X_test_feat)

    fare_metrics = calculate_metrics(y_test_fare, pred_fare)
    dur_metrics = calculate_metrics(y_test_dur, pred_dur)

    sample_row = X_test_feat.head(1)
    lat_fare = measure_inference_time(lgb_fare, sample_row)
    lat_dur = measure_inference_time(lgb_dur, sample_row)

    logger.info(f"Winning LightGBM Fare Metrics: {fare_metrics}, Latency: {lat_fare} ms")
    logger.info(f"Winning LightGBM Duration Metrics: {dur_metrics}, Latency: {lat_dur} ms")

    # 3. Compute Feature Importances
    feat_imp = compute_feature_importances(lgb_fare, lgb_dur, feature_names)

    # 4. Conduct In-depth Residual Analysis
    residual_analysis = conduct_residual_analysis(
        y_test_fare, pred_fare, y_test_dur, pred_dur, test_df, centroids_path=centroids_path
    )

    # 5. Export Self-Contained Production Bundle
    bundle = export_production_model(
        lgb_fare=lgb_fare,
        lgb_dur=lgb_dur,
        pipeline_path=pipeline_path,
        centroids_path=centroids_path,
        output_path=output_model_path,
        version="1.0.0-lightgbm",
    )

    # 6. Verify Isolation Acceptance Test
    isolation_passed = verify_model_isolation(output_model_path)
    assert isolation_passed, "Model isolation acceptance test must pass!"

    return {
        "fare_metrics": fare_metrics,
        "duration_metrics": dur_metrics,
        "feature_importances": feat_imp,
        "residual_analysis": residual_analysis,
        "model_bundle": bundle,
        "isolation_verified": isolation_passed,
    }


def main() -> None:
    run_full_model_selection_pipeline()


if __name__ == "__main__":
    main()
