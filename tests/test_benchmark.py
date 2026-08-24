# tests/test_benchmark.py — Automated tests for API benchmarking & latency optimization (T-113).

from pathlib import Path

import pandas as pd
import pytest

from scripts.benchmark_api import (
    LATENCY_CONCURRENCY,
    SAMPLE_PAYLOADS,
    SLA_BUDGET,
    run_in_process_benchmark,
)

pytestmark = pytest.mark.requires_model


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_predict_fast_parity_with_dataframe_predict(repo_root):
    """Assert that predict_fast produces identical predictions to standard DataFrame predict.

    Path-independent parity is covered by tests/test_predictor_parity.py, which needs no
    artifact. This one adds the real bundle's centroids and target encodings on top.
    """
    import pickle

    model_path = repo_root / "models" / "model.pkl"

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
    """Verify that run_in_process_benchmark runs and produces coherent metrics.

    Deliberately asserts no absolute timing. An earlier version required
    `throughput_rps > 100`, which turned any loaded machine — a busy laptop, a shared
    CI runner — into a red suite for reasons unrelated to the code under test. Absolute
    performance belongs in the benchmark report, not in a correctness suite.
    """
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
    assert result.throughput_rps > 0.0

    # Percentiles must be ordered and positive whatever the machine was doing.
    assert 0.0 < result.latency_p50_ms <= result.latency_p95_ms <= result.latency_p99_ms
    assert result.latency_min_ms <= result.latency_p50_ms <= result.latency_max_ms

    assert "fast_predict_total_ms" in result.component_breakdown_ms
    assert "lightgbm_inference_ms" in result.component_breakdown_ms


def test_latency_sla_is_only_judged_where_it_is_meaningful():
    """The latency budget applies to service time, not to queueing under saturation."""
    saturated = run_in_process_benchmark(
        scenario_name="Saturated",
        total_requests=60,
        concurrency=LATENCY_CONCURRENCY + 4,
        use_fast_path=True,
        warm_up=True,
    )

    assert not saturated.measures_latency
    assert saturated.sla_latency_passed is False, (
        "A saturated run must never report a latency pass: its percentiles measure how "
        "long requests waited for a worker, which the model cannot influence."
    )


def test_throughput_sla_is_actually_evaluated():
    """min_throughput_rps used to be declared in SLA_BUDGET and never checked."""
    assert "min_throughput_rps" in SLA_BUDGET

    result = run_in_process_benchmark(
        scenario_name="Throughput",
        total_requests=100,
        concurrency=5,
        use_fast_path=True,
        warm_up=True,
    )

    expected = result.throughput_rps >= SLA_BUDGET["min_throughput_rps"]
    assert result.sla_throughput_passed is expected


def test_warm_up_method_functional(repo_root):
    """Verify that model warm_up executes without errors."""
    import pickle

    model_path = repo_root / "models" / "model.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    assert hasattr(model, "warm_up")
    model.warm_up()
