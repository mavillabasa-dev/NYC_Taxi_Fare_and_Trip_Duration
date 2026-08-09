"""Tests for the T-107 gradient-boosting experiment layer."""

import json

import numpy as np
import pandas as pd
import pytest

from src.gradient_boosting import (
    SearchConfig,
    regression_metrics,
    run_experiments,
    save_results,
    validate_feature_contract,
)


@pytest.fixture
def numeric_temporal_splits():
    rng = np.random.default_rng(42)
    rows = 72
    distance = np.linspace(0.5, 18, rows)
    frame = pd.DataFrame(
        {
            "trip_distance": distance,
            "hour": np.arange(rows) % 24,
            "PULocationID": rng.integers(1, 266, rows),
            "fare_amount": 3.0 + 2.7 * distance + rng.normal(0, 0.2, rows),
            "duration_minutes": 4.0 + 3.5 * distance + rng.normal(0, 0.3, rows),
        }
    )
    return frame.iloc[:56].copy(), frame.iloc[56:].copy()


def test_feature_contract_rejects_target_and_post_trip_columns():
    with pytest.raises(ValueError, match="Leakage columns"):
        validate_feature_contract(pd.DataFrame({"fare_amount": [10], "tip_amount": [2]}))


def test_regression_metrics_are_reported_in_original_units():
    metrics = regression_metrics(pd.Series([10.0, 20.0]), np.array([12.0, 18.0]))
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["mape_pct"] == pytest.approx(15.0)
    assert metrics["r2"] == pytest.approx(0.84)


def test_overlapping_temporal_splits_are_rejected():
    train_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2022-05-20", "2022-05-25"]),
            "trip_distance": [1.0, 2.0],
            "fare_amount": [8.0, 12.0],
            "duration_minutes": [10.0, 18.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2022-05-23"]),
            "trip_distance": [1.5],
            "fare_amount": [10.0],
            "duration_minutes": [14.0],
        }
    )

    with pytest.raises(ValueError, match="periods overlap"):
        run_experiments(train_df, test_df)


@pytest.mark.parametrize("model_family", ["lightgbm", "xgboost"])
def test_experiment_trains_both_targets_and_saves_report(
    model_family, numeric_temporal_splits, tmp_path
):
    train_df, test_df = numeric_temporal_splits
    config = SearchConfig(n_iter=1, cv_splits=2, n_jobs=1, verbose=0)

    runs = run_experiments(
        train_df,
        test_df,
        targets=("fare_amount", "duration_minutes"),
        model_families=(model_family,),
        config=config,
    )

    assert {run.target for run in runs} == {"fare_amount", "duration_minutes"}
    assert all(run.metrics.keys() == {"mae", "rmse", "mape_pct", "r2"} for run in runs)
    assert all(run.training_seconds > 0 for run in runs)
    assert all(run.inference_ms_per_row > 0 for run in runs)
    assert all(run.feature_importance for run in runs)
    assert all(sum(run.feature_importance.values()) == pytest.approx(1.0) for run in runs)

    save_results(runs, tmp_path)
    report = json.loads((tmp_path / "t107_results.json").read_text())
    assert report["ticket"] == "T-107"
    assert len(report["runs"]) == 2
    assert (tmp_path / f"t107_{model_family}_fare_amount.joblib").exists()
