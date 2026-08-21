# tests/test_benchmark.py — Automated tests for API benchmarking & latency optimization (T-113).

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_api import (
    SAMPLE_PAYLOADS,
    SLA_BUDGET,
    run_in_process_benchmark,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_predict_fast_parity_with_dataframe_predict(repo_root):
    """Assert that predict_fast produces identical predictions to standard DataFrame predict."""
    import pickle

    model_path = repo_root / "models" / "model.pkl"
    assert model_path.exists(), "models/model.pkl must exist"

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    assert hasattr(model, "predict_fast"), "SelfContainedTaxiModel must have predict_fast method"

    for payload in SAMPLE_PAYLOADS:
        # Standard DataFrame predict
        df = pd.DataFrame([payload])
        standard_pred = model.predict(df)[0]
        std_fare, std_dur = float(standard_pred[0]), float(standard_pred[1])

        # Fast scalar predict
        fast_fare, fast_dur = model.predict_fast(payload)

        # Assert parity within numerical tolerance ($0.05 / 0.1 min due to float math)
        assert abs(fast_fare - std_fare) < 0.05, f"Fare mismatch: fast={fast_fare}, std={std_fare}"
        assert abs(fast_dur - std_dur) < 0.1, f"Duration mismatch: fast={fast_dur}, std={std_dur}"


def test_benchmark_in_process_execution():
    """Verify that run_in_process_benchmark runs successfully and produces complete metrics."""
    result = run_in_process_benchmark(
        scenario_name="Test Benchmark",
        total_requests=100,
        concurrency=5,
        use_fast_path=True,
        warm_up=True,
    )

    assert result.total_requests == 100
    assert result.success_count == 100
    assert result.error_count == 0
    assert result.throughput_rps > 100.0, f"Expected >100 req/s, got {result.throughput_rps}"
    assert result.latency_p50_ms > 0.0
    assert result.latency_p95_ms >= result.latency_p50_ms
    assert result.latency_p99_ms >= result.latency_p95_ms
    assert "fast_predict_total_ms" in result.component_breakdown_ms
    assert "lightgbm_inference_ms" in result.component_breakdown_ms


def test_warm_up_method_functional(repo_root):
    """Verify that model warm_up executes without errors."""
    import pickle

    model_path = repo_root / "models" / "model.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    assert hasattr(model, "warm_up")
    model.warm_up()
