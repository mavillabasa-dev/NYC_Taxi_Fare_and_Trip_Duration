# tests/test_baselines.py — Unit tests for baseline regressors and metrics (T-106)
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ALLOWED_FEATURES,
    MODELS_DIR,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.features import NYCFeaturePipeline
from src.train import (
    calculate_metrics,
    measure_inference_time,
    train_and_evaluate_baselines,
)


def test_calculate_metrics():
    """Test mathematical correctness of MAE, RMSE, MAPE, and R2 calculations."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "R2" in metrics

    assert metrics["MAE"] == 2.5
    assert metrics["R2"] > 0.9


def test_measure_inference_time():
    """Test single-row inference latency measurement."""
    X_dummy = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})
    y_dummy = np.array([10.0, 20.0, 30.0])

    model = DummyRegressor(strategy="mean")
    model.fit(X_dummy, y_dummy)

    lat_ms = measure_inference_time(model, X_dummy.head(1), n_trials=100)
    assert isinstance(lat_ms, float)
    assert lat_ms >= 0.0


def test_dummy_regressor_mean_prediction():
    """Test that DummyRegressor predicts exact training mean on test data."""
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
    y_train = np.array([10.0, 20.0, 30.0])  # mean = 20.0

    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({"feat": [10.0, 20.0]})
    preds = model.predict(X_test)

    np.testing.assert_array_almost_equal(preds, np.array([20.0, 20.0]))


def test_decision_tree_baseline_fit_and_predict():
    """Test Decision Tree training and performance on synthetic dataset."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y_train = X_train["f0"] * 5.0 + X_train["f1"] * 2.0 + 10.0

    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
    y_test = X_test["f0"] * 5.0 + X_test["f1"] * 2.0 + 10.0

    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    assert len(preds) == 20

    metrics = calculate_metrics(y_test.values, preds)
    assert metrics["R2"] > 0.7  # Decision tree captures strong linear combination


def test_train_and_evaluate_baselines_integration():
    """Integration test running baseline pipeline on actual cleaned parquet files."""
    if not (os.path.exists(TRAIN_CLEANED_PATH) and os.path.exists(TEST_CLEANED_PATH)):
        pytest.skip("Cleaned parquet files not found")

    output = train_and_evaluate_baselines(save_models=False)
    results = output["results"]

    assert "Trivial_Mean_Fare" in results
    assert "Trivial_Mean_Duration" in results
    assert "DecisionTree_Fare" in results
    assert "DecisionTree_Duration" in results

    # Assert Decision Tree outperforms Trivial Mean on test set
    assert results["DecisionTree_Fare"]["MAE"] < results["Trivial_Mean_Fare"]["MAE"]
    assert results["DecisionTree_Fare"]["R2"] > results["Trivial_Mean_Fare"]["R2"]
    assert results["DecisionTree_Duration"]["MAE"] < results["Trivial_Mean_Duration"]["MAE"]
    assert results["DecisionTree_Duration"]["R2"] > results["Trivial_Mean_Duration"]["R2"]
