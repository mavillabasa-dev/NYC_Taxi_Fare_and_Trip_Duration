# tests/test_mlp.py — Unit tests for Multi-Layer Perceptron (MLP) Regressors (T-108)
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

from src.config import RANDOM_SEED
from src.mlp import (
    MLPConfig,
    build_mlp_pipeline,
    calculate_metrics,
    measure_single_row_latency,
)


def test_build_mlp_pipeline_structure():
    """Test that build_mlp_pipeline creates an sklearn Pipeline with scaling and MLP steps."""
    config = MLPConfig(hidden_layer_sizes=(64, 32), max_iter=10)
    pipeline = build_mlp_pipeline(config)

    assert isinstance(pipeline, Pipeline)
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps
    assert "mlp" in pipeline.named_steps

    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["mlp"], MLPRegressor)

    mlp_step = pipeline.named_steps["mlp"]
    assert mlp_step.hidden_layer_sizes == (64, 32)
    assert mlp_step.early_stopping is True
    assert mlp_step.validation_fraction == 0.1


def test_mlp_pipeline_fit_and_predict_synthetic():
    """Test MLP fitting and prediction on synthetic dataset with missing values."""
    np.random.seed(RANDOM_SEED)
    X_train = pd.DataFrame(np.random.randn(500, 6), columns=[f"f{i}" for i in range(6)])
    # Insert NaN to test imputer inside pipeline
    X_train.iloc[0, 0] = np.nan
    X_train.iloc[5, 2] = np.nan
    y_train = X_train["f1"].fillna(0) * 10.0 + X_train["f2"].fillna(0) * 5.0 + 20.0

    X_test = pd.DataFrame(np.random.randn(100, 6), columns=[f"f{i}" for i in range(6)])
    X_test.iloc[2, 0] = np.nan
    y_test = X_test["f1"].fillna(0) * 10.0 + X_test["f2"].fillna(0) * 5.0 + 20.0

    config = MLPConfig(
        hidden_layer_sizes=(32, 16), max_iter=60, batch_size=64, learning_rate_init=0.01
    )
    pipeline = build_mlp_pipeline(config)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    assert len(preds) == 100
    assert not np.isnan(preds).any()

    metrics = calculate_metrics(y_test.values, preds)
    assert "MAE" in metrics
    assert "R2" in metrics
    assert isinstance(metrics["R2"], float)
    assert metrics["MAE"] > 0.0


def test_mlp_single_row_latency_measurement():
    """Test warm single-row inference latency measurement."""
    np.random.seed(RANDOM_SEED)
    X = pd.DataFrame(np.random.randn(50, 2), columns=["f1", "f2"])
    y = np.random.randn(50) * 10.0 + 20.0

    config = MLPConfig(hidden_layer_sizes=(16,), max_iter=10, batch_size=16)
    pipeline = build_mlp_pipeline(config)
    pipeline.fit(X, y)

    lat_ms = measure_single_row_latency(pipeline, X.head(1), n_trials=50)
    assert isinstance(lat_ms, float)
    assert lat_ms >= 0.0


def test_mlp_early_stopping_behavior():
    """Test that early stopping records validation scores on training validation slice."""
    np.random.seed(RANDOM_SEED)
    X = pd.DataFrame(np.random.randn(300, 4), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] * 5.0 + 2.0

    config = MLPConfig(hidden_layer_sizes=(32,), max_iter=30, batch_size=32)
    pipeline = build_mlp_pipeline(config)
    pipeline.fit(X, y)

    mlp_estimator: MLPRegressor = pipeline.named_steps["mlp"]
    assert hasattr(mlp_estimator, "loss_curve_")
    assert len(mlp_estimator.loss_curve_) > 0
    assert hasattr(mlp_estimator, "validation_scores_")
