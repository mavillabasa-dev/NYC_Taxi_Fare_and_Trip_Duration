"""src/mlp.py — Multi-Layer Perceptron (MLP) Regressor Evaluation (T-108).

This module implements the training, benchmarking, and evaluation of:
Multi-Layer Perceptron (MLP) Neural Networks for NYC Taxi fare amount and trip duration.

SINGLE-OUTPUT ARCHITECTURE & PIPELINE SCALING (T-108):
------------------------------------------------------
1. Architecture: Two independent single-output MLP neural networks (one for fare_amount,
   one for duration_minutes) adhering to the T-106 architectural decision.
2. Pipeline Scaling: Features are imputed (SimpleImputer) and standard scaled (StandardScaler)
   strictly inside the scikit-learn Pipeline, ensuring no ad-hoc feature scaling.
3. Early Stopping: Validated on an internal 10% slice carved strictly from the training split.
4. Metrics: MAE, RMSE, MAPE, R² in original target units ($ and minutes), plus training time
   and warm single-row inference latency measurements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMNS,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.features import NYCFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    """Hyperparameter configuration for Multi-Layer Perceptron."""

    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: int = 4096
    learning_rate_init: float = 0.002
    max_iter: int = 20
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 3
    tol: float = 1e-4
    random_seed: int = RANDOM_SEED


@dataclass
class MLPRunResult:
    """Evaluation results and metadata for an MLP model run."""

    target: str
    architecture: str
    optimizer: str
    metrics: Dict[str, float]
    training_seconds: float
    inference_ms_per_row: float
    loss_curve: List[float]
    validation_scores: List[float]
    epochs_trained: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_mlp_pipeline(config: MLPConfig = MLPConfig()) -> Pipeline:
    """Builds an sklearn Pipeline comprising imputer, standard scaler, and MLP regressor."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation=config.activation,
                    solver=config.solver,
                    alpha=config.alpha,
                    batch_size=config.batch_size,
                    learning_rate_init=config.learning_rate_init,
                    max_iter=config.max_iter,
                    early_stopping=config.early_stopping,
                    validation_fraction=config.validation_fraction,
                    n_iter_no_change=config.n_iter_no_change,
                    tol=config.tol,
                    random_state=config.random_seed,
                    verbose=False,
                ),
            ),
        ]
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, MAPE (%), and R2 in original target units."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
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


def measure_single_row_latency(
    estimator: Pipeline, sample_row: pd.DataFrame, n_trials: int = 1000
) -> float:
    """Measures single-row inference latency in milliseconds."""
    # Warm-up
    _ = estimator.predict(sample_row)

    start_time = time.perf_counter()
    for _ in range(n_trials):
        _ = estimator.predict(sample_row)
    elapsed = time.perf_counter() - start_time

    return round((elapsed / n_trials) * 1000.0, 4)


def train_and_evaluate_mlp(
    train_path: str = TRAIN_CLEANED_PATH,
    test_path: str = TEST_CLEANED_PATH,
    config: MLPConfig = MLPConfig(),
    save_models: bool = True,
    output_dir: str = os.path.join(MODELS_DIR, "t108"),
) -> Dict[str, Any]:
    """Trains and evaluates MLP models for fare_amount and duration_minutes."""
    logger.info("--- Starting Multi-Layer Perceptron (MLP) Evaluation (T-108) ---")

    logger.info(f"Loading cleaned train data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading cleaned test data from {test_path}...")
    test_df = pd.read_parquet(test_path)

    # 1. Feature Engineering
    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
    if os.path.exists(pipeline_path):
        logger.info(f"Loading existing feature pipeline from {pipeline_path}...")
        with open(pipeline_path, "rb") as f:
            feature_pipeline = pickle.load(f)
        X_train_feat = feature_pipeline.transform(train_df[ALLOWED_FEATURES])
    else:
        logger.info("Fitting new NYCFeaturePipeline on X_train, y_train...")
        feature_pipeline = NYCFeaturePipeline()
        X_train_feat = feature_pipeline.fit_transform(
            train_df[ALLOWED_FEATURES],
            train_df[["fare_amount", "duration_minutes"]],
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(pipeline_path, "wb") as f:
            pickle.dump(feature_pipeline, f)

    X_test_feat = feature_pipeline.transform(test_df[ALLOWED_FEATURES])

    sample_single_row = X_test_feat.head(1)
    results = {}
    fitted_pipelines = {}
    reports = {}

    for target in TARGET_COLUMNS:
        logger.info(f"Training MLP for target: {target} (architecture={config.hidden_layer_sizes})...")
        y_train = train_df[target].values
        y_test = test_df[target].values

        mlp_pipe = build_mlp_pipeline(config)

        t0 = time.time()
        mlp_pipe.fit(X_train_feat, y_train)
        train_time_sec = time.time() - t0

        y_pred = mlp_pipe.predict(X_test_feat)
        metrics = calculate_metrics(y_test, y_pred)
        latency_ms = measure_single_row_latency(mlp_pipe, sample_single_row)

        mlp_model: MLPRegressor = mlp_pipe.named_steps["mlp"]
        loss_curve = [float(x) for x in getattr(mlp_model, "loss_curve_", [])]
        val_scores = [
            float(x) for x in getattr(mlp_model, "validation_scores_", [])
        ]
        n_epochs = int(getattr(mlp_model, "n_iter_", 0))

        run_res = MLPRunResult(
            target=target,
            architecture=f"MLP{config.hidden_layer_sizes}-ReLU",
            optimizer=f"Adam(lr={config.learning_rate_init}, batch={config.batch_size})",
            metrics=metrics,
            training_seconds=round(train_time_sec, 4),
            inference_ms_per_row=latency_ms,
            loss_curve=loss_curve,
            validation_scores=val_scores,
            epochs_trained=n_epochs,
        )

        results[f"MLP_{target}"] = {
            **metrics,
            "train_time_sec": round(train_time_sec, 4),
            "inference_latency_ms": latency_ms,
        }
        fitted_pipelines[target] = mlp_pipe
        reports[target] = run_res.to_dict()

        logger.info(
            f"Finished MLP for {target}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, "
            f"R2={metrics['R2']}, Time={train_time_sec:.2f}s, Latency={latency_ms:.4f}ms"
        )

    summary_df = pd.DataFrame(results).T
    logger.info("\n=== MLP Evaluation Summary (Test Set) ===\n" + summary_df.to_string())

    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        for target, pipe in fitted_pipelines.items():
            model_file = os.path.join(output_dir, f"t108_mlp_{target}.joblib")
            joblib.dump(pipe, model_file)
            logger.info(f"Saved {target} MLP pipeline to {model_file}")

        results_json_path = os.path.join(output_dir, "t108_mlp_results.json")
        with open(results_json_path, "w") as f:
            json.dump(reports, f, indent=2)
        logger.info(f"Saved MLP metrics report to {results_json_path}")

        bundle_path = os.path.join(MODELS_DIR, "mlp_models.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump(fitted_pipelines, f)
        logger.info(f"Saved MLP model bundle to {bundle_path}")

    return {
        "results": results,
        "summary_table": summary_df,
        "models": fitted_pipelines,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T-108 MLP regression experiments")
    parser.add_argument("--train", default=TRAIN_CLEANED_PATH)
    parser.add_argument("--test", default=TEST_CLEANED_PATH)
    parser.add_argument("--output-dir", default=os.path.join(MODELS_DIR, "t108"))
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    config = MLPConfig(
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_rate_init=args.lr,
    )
    train_and_evaluate_mlp(
        train_path=args.train,
        test_path=args.test,
        config=config,
        save_models=True,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
