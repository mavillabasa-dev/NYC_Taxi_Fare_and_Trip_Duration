"""src/train.py — Baseline Model Training & Evaluation (T-106).

This module implements the training, benchmarking, and evaluation of:
1. Trivial Baseline Regressor (Predicting training mean for each target)
2. Decision Tree Regressor (For both fare_amount and duration_minutes)

SINGLE-OUTPUT VS. MULTI-OUTPUT MODEL ARCHITECTURE DECISION (T-106):
------------------------------------------------------------------
DECISION: We train two separate single-output models (one for fare_amount, one for duration_minutes)
rather than a single multi-output model.
JUSTIFICATION:
1. Fares ($) and durations (minutes) have distinct loss surfaces, target scales, and error distributions.
2. Independent single-output models allow independent hyperparameter optimization (e.g. tree depth,
   min samples per leaf, regularisation) tailored to each target.
3. Feature importance and residual diagnostics remain clear and decoupled per target.
This choice binds T-107 (GBDTs), T-108 (MLP), T-109 (Model Selection), and T-110 (API contract).
"""

import logging
import os
import pickle
import time
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.features import NYCFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates regression metrics: MAE, RMSE, MAPE, and R2 score.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Calculate MAPE excluding zero denominators safely
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }


def measure_inference_time(model: Any, sample_row: pd.DataFrame, n_trials: int = 1000) -> float:
    """
    Measures average single-row inference latency in milliseconds across n_trials.
    """
    # Warmup call
    _ = model.predict(sample_row)

    start_time = time.perf_counter()
    for _ in range(n_trials):
        _ = model.predict(sample_row)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / n_trials) * 1000.0
    return round(avg_ms, 4)


def train_and_evaluate_baselines(
    train_path: str = TRAIN_CLEANED_PATH,
    test_path: str = TEST_CLEANED_PATH,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    Trains and benchmarks the Trivial Mean Baseline and Decision Tree Regressors
    on both fare_amount and duration_minutes targets.
    """
    logger.info("--- Starting Baseline Model Training (T-106) ---")

    logger.info(f"Loading cleaned train data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading cleaned test data from {test_path}...")
    test_df = pd.read_parquet(test_path)

    # 1. Feature Engineering
    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
    if os.path.exists(pipeline_path):
        logger.info(f"Loading existing feature pipeline from {pipeline_path}...")
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
        X_train_feat = pipeline.transform(train_df[ALLOWED_FEATURES])
    else:
        logger.info("Fitting new NYCFeaturePipeline on X_train, y_train...")
        pipeline = NYCFeaturePipeline()
        X_train_feat = pipeline.fit_transform(
            train_df[ALLOWED_FEATURES],
            train_df[["fare_amount", "duration_minutes"]],
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

    X_test_feat = pipeline.transform(test_df[ALLOWED_FEATURES])

    y_train_fare = train_df["fare_amount"].values
    y_test_fare = test_df["fare_amount"].values

    y_train_dur = train_df["duration_minutes"].values
    y_test_dur = test_df["duration_minutes"].values

    sample_single_row = X_test_feat.head(1)

    results = {}

    # 2. Trivial Mean Baseline
    logger.info("--- Training Trivial Mean Baseline ---")

    # Fare Trivial Mean
    t0 = time.time()
    dummy_fare = DummyRegressor(strategy="mean")
    dummy_fare.fit(X_train_feat, y_train_fare)
    t_train_dummy_fare = time.time() - t0
    y_pred_dummy_fare = dummy_fare.predict(X_test_feat)
    metrics_dummy_fare = calculate_metrics(y_test_fare, y_pred_dummy_fare)
    lat_dummy_fare = measure_inference_time(dummy_fare, sample_single_row)

    # Duration Trivial Mean
    t0 = time.time()
    dummy_dur = DummyRegressor(strategy="mean")
    dummy_dur.fit(X_train_feat, y_train_dur)
    t_train_dummy_dur = time.time() - t0
    y_pred_dummy_dur = dummy_dur.predict(X_test_feat)
    metrics_dummy_dur = calculate_metrics(y_test_dur, y_pred_dummy_dur)
    lat_dummy_dur = measure_inference_time(dummy_dur, sample_single_row)

    results["Trivial_Mean_Fare"] = {
        **metrics_dummy_fare,
        "train_time_sec": round(t_train_dummy_fare, 4),
        "inference_latency_ms": lat_dummy_fare,
    }
    results["Trivial_Mean_Duration"] = {
        **metrics_dummy_dur,
        "train_time_sec": round(t_train_dummy_dur, 4),
        "inference_latency_ms": lat_dummy_dur,
    }

    # 3. Decision Tree Baseline
    logger.info("--- Training Decision Tree Regressors ---")

    # Fare Decision Tree
    t0 = time.time()
    dt_fare = DecisionTreeRegressor(max_depth=12, min_samples_leaf=50, random_state=RANDOM_SEED)
    dt_fare.fit(X_train_feat, y_train_fare)
    t_train_dt_fare = time.time() - t0
    y_pred_dt_fare = dt_fare.predict(X_test_feat)
    metrics_dt_fare = calculate_metrics(y_test_fare, y_pred_dt_fare)
    lat_dt_fare = measure_inference_time(dt_fare, sample_single_row)

    # Duration Decision Tree
    t0 = time.time()
    dt_dur = DecisionTreeRegressor(max_depth=12, min_samples_leaf=50, random_state=RANDOM_SEED)
    dt_dur.fit(X_train_feat, y_train_dur)
    t_train_dt_dur = time.time() - t0
    y_pred_dt_dur = dt_dur.predict(X_test_feat)
    metrics_dt_dur = calculate_metrics(y_test_dur, y_pred_dt_dur)
    lat_dt_dur = measure_inference_time(dt_dur, sample_single_row)

    results["DecisionTree_Fare"] = {
        **metrics_dt_fare,
        "train_time_sec": round(t_train_dt_fare, 4),
        "inference_latency_ms": lat_dt_fare,
    }
    results["DecisionTree_Duration"] = {
        **metrics_dt_dur,
        "train_time_sec": round(t_train_dt_dur, 4),
        "inference_latency_ms": lat_dt_dur,
    }

    # Print summary table
    summary_df = pd.DataFrame(results).T
    logger.info("\n=== Baseline Models Evaluation Summary (Test Set) ===\n" + summary_df.to_string())

    if save_models:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "dummy_models.pkl"), "wb") as f:
            pickle.dump({"fare": dummy_fare, "duration": dummy_dur}, f)
        with open(os.path.join(MODELS_DIR, "dt_models.pkl"), "wb") as f:
            pickle.dump({"fare": dt_fare, "duration": dt_dur}, f)
        logger.info(f"Saved baseline model artifacts to {MODELS_DIR}")

    return {
        "results": results,
        "summary_table": summary_df,
        "models": {
            "dummy_fare": dummy_fare,
            "dummy_dur": dummy_dur,
            "dt_fare": dt_fare,
            "dt_dur": dt_dur,
        },
    }


if __name__ == "__main__":
    train_and_evaluate_baselines()
