"""Reproducible LightGBM and XGBoost experiments for T-107.

This module deliberately does not implement feature engineering. It accepts the
transformer template produced by T-105 and clones it into every candidate pipeline,
ensuring that preprocessing is fitted inside each temporal CV fold.
``trip_distance`` is an accepted approximation: at serving time it must be a
routing estimate rather than the completed trip's metered distance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import (
    BANNED_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMNS,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)

logger = logging.getLogger(__name__)

MODEL_FAMILIES = ("lightgbm", "xgboost")
SUSPICIOUS_IMPORTANCE_THRESHOLD = 0.65

# Small, explicit search spaces make the experiment auditable and reasonably
# affordable on the 3M-row May dataset. RandomizedSearchCV samples ``n_iter``
# configurations from each space using RANDOM_SEED.
SEARCH_SPACES: Mapping[str, Mapping[str, Sequence[Any]]] = {
    "lightgbm": {
        "n_estimators": (200, 400, 700),
        "learning_rate": (0.03, 0.05, 0.1),
        "num_leaves": (31, 63, 127),
        "max_depth": (-1, 8, 12),
        "min_child_samples": (20, 50, 100),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8, 1.0),
        "reg_lambda": (0.0, 1.0, 5.0),
    },
    "xgboost": {
        "n_estimators": (200, 400, 700),
        "learning_rate": (0.03, 0.05, 0.1),
        "max_depth": (4, 6, 8),
        "min_child_weight": (1, 5, 10),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8, 1.0),
        "reg_lambda": (1.0, 5.0, 10.0),
    },
}


@dataclass(frozen=True)
class SearchConfig:
    """Hyperparameter-search budget and temporal CV configuration."""

    n_iter: int = 12
    cv_splits: int = 3
    scoring: str = "neg_mean_absolute_error"
    n_jobs: int = -1
    verbose: int = 1
    random_seed: int = RANDOM_SEED


@dataclass
class ModelRun:
    """Results and fitted estimator for one model family/target pair."""

    model_family: str
    target: str
    best_params: Dict[str, Any]
    cv_mae: float
    metrics: Dict[str, float]
    training_seconds: float
    inference_ms_per_row: float
    feature_importance: Dict[str, float]
    suspicious_features: Sequence[str]
    estimator: BaseEstimator = field(repr=False)

    def report_dict(self) -> Dict[str, Any]:
        """Return the JSON-serializable portion of the run."""

        return {
            "model_family": self.model_family,
            "target": self.target,
            "best_params": self.best_params,
            "cv_mae": self.cv_mae,
            "metrics": self.metrics,
            "training_seconds": self.training_seconds,
            "inference_ms_per_row": self.inference_ms_per_row,
            "feature_importance": self.feature_importance,
            "suspicious_features": list(self.suspicious_features),
        }


def validate_feature_contract(features: pd.DataFrame) -> None:
    """Reject targets or known post-trip columns before a model can see them."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    forbidden = sorted(set(features.columns) & set(BANNED_COLUMNS + TARGET_COLUMNS))
    if forbidden:
        raise ValueError(f"Leakage columns found in model features: {forbidden}")


def _make_regressor(model_family: str, random_seed: int) -> BaseEstimator:
    if model_family == "lightgbm":
        return LGBMRegressor(
            objective="regression_l1",
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            importance_type="gain",
            subsample_freq=1,
        )
    if model_family == "xgboost":
        return XGBRegressor(
            objective="reg:absoluteerror",
            random_state=random_seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            importance_type="gain",
        )
    raise ValueError(f"Unknown model family {model_family!r}; use {MODEL_FAMILIES}")


def _pipeline_and_space(
    model_family: str,
    transformer: Optional[TransformerMixin],
    random_seed: int,
) -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    model = _make_regressor(model_family, random_seed)
    if transformer is None:
        return model, dict(SEARCH_SPACES[model_family])

    pipeline = Pipeline(
        [("features", clone(transformer)), ("model", model)]
    )
    space = {f"model__{key}": value for key, value in SEARCH_SPACES[model_family].items()}
    return pipeline, space


def tune_model(
    model_family: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    transformer: Optional[TransformerMixin] = None,
    config: SearchConfig = SearchConfig(),
) -> Tuple[BaseEstimator, Dict[str, Any], float, float]:
    """Tune one regressor with forward-only temporal cross-validation."""

    validate_feature_contract(x_train)
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must contain the same number of rows")
    if len(x_train) <= config.cv_splits:
        raise ValueError("Training rows must exceed the number of CV splits")

    estimator, parameters = _pipeline_and_space(
        model_family, transformer, config.random_seed
    )
    cv = TimeSeriesSplit(n_splits=config.cv_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=config.n_iter,
        scoring=config.scoring,
        cv=cv,
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        return_train_score=False,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    cv_mae = -float(search.best_score_)
    return search.best_estimator_, dict(search.best_params_), cv_mae, elapsed


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
    """Calculate the common T-106/T-107 metric set in original target units."""

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mape_pct": float(mean_absolute_percentage_error(y_true, predictions) * 100),
        "r2": float(r2_score(y_true, predictions)),
    }


def measure_single_row_latency(
    estimator: BaseEstimator, x_test: pd.DataFrame, repetitions: int = 30
) -> float:
    """Measure warm single-row inference latency in milliseconds."""

    if x_test.empty:
        raise ValueError("x_test cannot be empty")
    row = x_test.iloc[[0]]
    estimator.predict(row)  # warm-up
    started = time.perf_counter()
    for _ in range(repetitions):
        estimator.predict(row)
    return (time.perf_counter() - started) * 1000 / repetitions


def _fitted_model(estimator: BaseEstimator) -> BaseEstimator:
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["model"]
    return estimator


def feature_importance(
    estimator: BaseEstimator, original_columns: Iterable[str]
) -> Dict[str, float]:
    """Return normalized gain/importances with transformed feature names when possible."""

    model = _fitted_model(estimator)
    values = np.asarray(model.feature_importances_, dtype=float)
    names = list(original_columns)
    if isinstance(estimator, Pipeline):
        transformer = estimator.named_steps["features"]
        try:
            names = list(transformer.get_feature_names_out())
        except (AttributeError, TypeError, ValueError):
            names = [f"feature_{index}" for index in range(len(values))]
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    total = values.sum()
    normalized = values / total if total > 0 else values
    ordered = sorted(zip(names, normalized), key=lambda item: item[1], reverse=True)
    return {str(name): float(value) for name, value in ordered}


def run_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transformer: Optional[TransformerMixin] = None,
    targets: Sequence[str] = tuple(TARGET_COLUMNS),
    model_families: Sequence[str] = MODEL_FAMILIES,
    config: SearchConfig = SearchConfig(),
) -> Sequence[ModelRun]:
    """Tune and evaluate LightGBM/XGBoost for every requested target."""

    missing = [target for target in targets if target not in train_df or target not in test_df]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    # Exclude both project targets even when a caller requests only one of them.
    feature_columns = [
        column for column in train_df.columns if column not in TARGET_COLUMNS
    ]
    if "tpep_pickup_datetime" in feature_columns:
        train_df = train_df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        test_df = test_df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        if (
            not train_df.empty
            and not test_df.empty
            and train_df["tpep_pickup_datetime"].max()
            >= test_df["tpep_pickup_datetime"].min()
        ):
            raise ValueError("Train/test periods overlap; expected a strict temporal split")
    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    validate_feature_contract(x_train)
    validate_feature_contract(x_test)

    runs = []
    for model_family in model_families:
        for target in targets:
            logger.info("Tuning %s for %s", model_family, target)
            estimator, params, cv_mae, training_seconds = tune_model(
                model_family,
                x_train,
                train_df[target],
                transformer=transformer,
                config=config,
            )
            predictions = estimator.predict(x_test)
            importances = feature_importance(estimator, feature_columns)
            suspicious = [
                name
                for name, importance in importances.items()
                if importance >= SUSPICIOUS_IMPORTANCE_THRESHOLD
            ]
            if suspicious:
                logger.warning(
                    "Dominant features for %s/%s require manual leakage review: %s",
                    model_family,
                    target,
                    suspicious,
                )
            runs.append(
                ModelRun(
                    model_family=model_family,
                    target=target,
                    best_params=params,
                    cv_mae=cv_mae,
                    metrics=regression_metrics(test_df[target], predictions),
                    training_seconds=training_seconds,
                    inference_ms_per_row=measure_single_row_latency(estimator, x_test),
                    feature_importance=importances,
                    suspicious_features=suspicious,
                    estimator=estimator,
                )
            )
    return runs


def save_results(runs: Sequence[ModelRun], output_dir: os.PathLike[str] | str) -> None:
    """Persist fitted candidate pipelines and a machine-readable comparison report."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for run in runs:
        joblib.dump(
            run.estimator,
            destination / f"t107_{run.model_family}_{run.target}.joblib",
        )
    report = {
        "ticket": "T-107",
        "search_method": "RandomizedSearchCV with TimeSeriesSplit",
        "library_versions": {
            "lightgbm": __import__("lightgbm").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
        "runs": [run.report_dict() for run in runs],
    }
    (destination / "t107_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T-107 boosting experiments")
    parser.add_argument("--train", default=TRAIN_CLEANED_PATH)
    parser.add_argument("--test", default=TEST_CLEANED_PATH)
    parser.add_argument(
        "--transformer",
        required=True,
        help="T-105 sklearn feature pipeline (models/feature_pipeline.pkl)",
    )
    parser.add_argument("--output-dir", default="models/t107")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    transformer = joblib.load(args.transformer)
    config = SearchConfig(
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    runs = run_experiments(train_df, test_df, transformer=transformer, config=config)
    save_results(runs, args.output_dir)
    logger.info("Saved %d candidate models and report to %s", len(runs), args.output_dir)


if __name__ == "__main__":
    main()
