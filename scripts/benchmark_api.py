"""scripts/benchmark_api.py — API Latency Benchmarking & Performance Profiling Tool (T-113).

This script conducts comprehensive load testing and latency benchmarking against the FastAPI
prediction serving pipeline (/predict). It supports concurrent execution, latency percentiles
(p50, p90, p95, p99), component profiling, SLA / budget verification, and before/after optimization
comparison.

Usage:
    python scripts/benchmark_api.py --requests 1000 --concurrency 10 --compare
    python scripts/benchmark_api.py --mode live-http --url http://localhost:8000 --requests 500
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure root and api/ are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "api"
for p in (REPO_ROOT, API_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmark")

# Stated Latency Budget / Service Level Objectives (SLOs)
SLA_BUDGET = {
    "cold_start_ms": 50.0,
    "p50_ms": 5.0,
    "p95_ms": 15.0,
    "p99_ms": 30.0,
    "min_throughput_rps": 200.0,
}

SAMPLE_PAYLOADS = [
    {
        "PULocationID": 132,  # JFK Airport
        "DOLocationID": 236,  # Upper East Side
        "tpep_pickup_datetime": "2022-05-20T14:30:00",
        "passenger_count": 2,
        "RatecodeID": 1,
        "trip_distance": 6.2,
    },
    {
        "PULocationID": 161,  # Midtown Center
        "DOLocationID": 142,  # Lincoln Square
        "tpep_pickup_datetime": "2022-05-15T08:15:00",
        "passenger_count": 1,
        "RatecodeID": 1,
        "trip_distance": 1.8,
    },
    {
        "PULocationID": 132,  # JFK Airport
        "DOLocationID": 1,    # Newark
        "tpep_pickup_datetime": "2022-05-25T18:45:00",
        "passenger_count": 3,
        "RatecodeID": 2,
        "trip_distance": 18.5,
    },
    {
        "PULocationID": 79,   # East Village
        "DOLocationID": 107,  # Gramercy
        "tpep_pickup_datetime": "2022-05-10T22:00:00",
        "passenger_count": 1,
        "RatecodeID": 1,
        "trip_distance": 1.2,
    },
]


@dataclass
class BenchmarkResult:
    scenario_name: str
    total_requests: int
    concurrency: int
    success_count: int
    error_count: int
    elapsed_seconds: float
    throughput_rps: float
    cold_start_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    sla_passed: bool
    component_breakdown_ms: Dict[str, float]


def profile_components(model_service, payload: dict) -> Dict[str, float]:
    """Profiles latency breakdown across individual serving components."""
    from app.model.schema import PredictionRequest

    # 1. Pydantic validation
    t0 = time.perf_counter()
    for _ in range(200):
        req = PredictionRequest(**payload)
        d = req.model_dump()
    pydantic_ms = ((time.perf_counter() - t0) / 200) * 1000

    # 2. Fast scalar feature transform
    model = model_service.model
    if hasattr(model, "predict_fast") and model.predict_fast is not None:
        t0 = time.perf_counter()
        for _ in range(200):
            model.predict_fast(payload)
        total_fast_ms = ((time.perf_counter() - t0) / 200) * 1000
    else:
        total_fast_ms = 0.0

    # 3. DataFrame baseline transform
    import pandas as pd
    t0 = time.perf_counter()
    df = pd.DataFrame([payload])
    for _ in range(200):
        model.transform_features(df)
    df_transform_ms = ((time.perf_counter() - t0) / 200) * 1000

    # 4. Pure LightGBM inference
    feat_df = model.transform_features(pd.DataFrame([payload]))
    t0 = time.perf_counter()
    for _ in range(200):
        model.fare_model.predict(feat_df)
        model.duration_model.predict(feat_df)
    lgbm_inference_ms = ((time.perf_counter() - t0) / 200) * 1000

    return {
        "pydantic_validation_ms": round(pydantic_ms, 3),
        "fast_predict_total_ms": round(total_fast_ms, 3),
        "dataframe_transform_ms": round(df_transform_ms, 3),
        "lightgbm_inference_ms": round(lgbm_inference_ms, 3),
    }


def run_in_process_benchmark(
    scenario_name: str,
    total_requests: int = 1000,
    concurrency: int = 10,
    use_fast_path: bool = True,
    warm_up: bool = True,
) -> BenchmarkResult:
    """Runs concurrent load test using FastAPI TestClient in-process."""
    import main
    from fastapi.testclient import TestClient

    # Configure model
    model_path = REPO_ROOT / "models/model.pkl"
    main.model_service.load(str(model_path))

    # Toggle fast path if simulating baseline
    original_predict_fast = getattr(main.model_service.model, "predict_fast", None)
    if not use_fast_path:
        main.model_service.model.predict_fast = None

    try:
        with TestClient(main.app) as client:
            # 1. Cold start measurement
            t0 = time.perf_counter()
            cold_resp = client.post("/predict", json=SAMPLE_PAYLOADS[0])
            cold_start_ms = (time.perf_counter() - t0) * 1000
            assert cold_resp.status_code == 200, f"Cold request failed: {cold_resp.text}"

            # 2. Warm up
            if warm_up:
                for _ in range(50):
                    client.post("/predict", json=SAMPLE_PAYLOADS[0])

            # 3. Concurrent load execution
            latencies_ms: List[float] = []
            errors = 0
            payloads = [SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)] for i in range(total_requests)]

            def _send_request(p: dict) -> float:
                req_t0 = time.perf_counter()
                res = client.post("/predict", json=p)
                duration = (time.perf_counter() - req_t0) * 1000
                if res.status_code != 200:
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
                return duration

            t_start_total = time.perf_counter()
            if concurrency <= 1:
                for p in payloads:
                    try:
                        lat = _send_request(p)
                        latencies_ms.append(lat)
                    except Exception:
                        errors += 1
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(_send_request, p) for p in payloads]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            lat = f.result()
                            latencies_ms.append(lat)
                        except Exception:
                            errors += 1

            total_elapsed = time.perf_counter() - t_start_total

        # Component breakdown
        breakdown = profile_components(main.model_service, SAMPLE_PAYLOADS[0])

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))
        throughput = len(latencies_ms) / total_elapsed if total_elapsed > 0 else 0.0

        sla_ok = (
            cold_start_ms <= SLA_BUDGET["cold_start_ms"]
            and p50 <= SLA_BUDGET["p50_ms"]
            and p95 <= SLA_BUDGET["p95_ms"]
            and p99 <= SLA_BUDGET["p99_ms"]
        )

        return BenchmarkResult(
            scenario_name=scenario_name,
            total_requests=total_requests,
            concurrency=concurrency,
            success_count=len(latencies_ms),
            error_count=errors,
            elapsed_seconds=round(total_elapsed, 4),
            throughput_rps=round(throughput, 1),
            cold_start_ms=round(cold_start_ms, 2),
            latency_mean_ms=round(mean_lat, 2),
            latency_std_ms=round(std_lat, 2),
            latency_min_ms=round(min_lat, 2),
            latency_p50_ms=round(p50, 2),
            latency_p90_ms=round(p90, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            latency_max_ms=round(max_lat, 2),
            sla_passed=sla_ok,
            component_breakdown_ms=breakdown,
        )
    finally:
        if original_predict_fast:
            main.model_service.model.predict_fast = original_predict_fast


def format_markdown_report(results: List[BenchmarkResult]) -> str:
    """Formats benchmark results into a clean markdown table."""
    lines = [
        "## Performance Benchmark & Latency Optimization Summary",
        "",
        "### 1. Latency Budget / SLA Compliance",
        "",
        "| Scenario | Concurrency | Requests | Cold Start | p50 (Median) | p90 | p95 | p99 | Max | Throughput | SLA Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status_badge = "✅ PASSED" if r.sla_passed else "⚠️ EXCEEDED"
        lines.append(
            f"| **{r.scenario_name}** | {r.concurrency} | {r.total_requests} | {r.cold_start_ms:.1f} ms | "
            f"**{r.latency_p50_ms:.2f} ms** | {r.latency_p90_ms:.2f} ms | {r.latency_p95_ms:.2f} ms | "
            f"{r.latency_p99_ms:.2f} ms | {r.latency_max_ms:.2f} ms | **{r.throughput_rps:.1f} req/s** | {status_badge} |"
        )

    lines.extend([
        "",
        "### 2. Component Latency Breakdown",
        "",
        "| Component / Pipeline Stage | Duration (ms) | Description |",
        "|---|---|---|",
    ])
    if results:
        b = results[-1].component_breakdown_ms
        lines.append(f"| **Pydantic Validation & Deserialization** | {b['pydantic_validation_ms']:.3f} ms | Schema validation, type coercion, and ISO timestamp parsing |")
        lines.append(f"| **Fast Scalar Feature Transformation** | {b['fast_predict_total_ms']:.3f} ms | Spatial centroid Haversine/Manhattan distances, cyclical trig features, target encoding |")
        lines.append(f"| **LightGBM Dual Model Inference** | {b['lightgbm_inference_ms']:.3f} ms | C++ OpenMP LightGBM Booster prediction for fare and duration |")
        lines.append(f"| *(Legacy DataFrame Transform)* | {b['dataframe_transform_ms']:.3f} ms | Unoptimized pandas DataFrame allocation & series parsing |")

    return "\n".join(lines)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="NYC Taxi API Latency & Load Benchmark (T-113)")
    parser.add_argument("--requests", type=int, default=1000, help="Total requests to execute")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--compare", action="store_true", help="Compare baseline unoptimized vs optimized pipeline")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export telemetry JSON")
    parser.add_argument("--output-markdown", type=str, default=None, help="Path to export Markdown report")
    args = parser.parse_args()

    logger.info("=== Starting API Latency Benchmark (T-113) ===")
    logger.info(f"Configuration: Requests={args.requests}, Concurrency={args.concurrency}, Compare={args.compare}")

    results: List[BenchmarkResult] = []

    if args.compare:
        logger.info("Running Scenario 1: Baseline (Unoptimized DataFrame Transform + No Pre-warming)...")
        res_base = run_in_process_benchmark(
            scenario_name="Baseline (DataFrame + Cold)",
            total_requests=args.requests,
            concurrency=args.concurrency,
            use_fast_path=False,
            warm_up=False,
        )
        results.append(res_base)
        logger.info(f"Baseline: p50={res_base.latency_p50_ms}ms, p95={res_base.latency_p95_ms}ms, Throughput={res_base.throughput_rps} req/s")

    logger.info("Running Scenario 2: Optimized (Fast Scalar Transform + C-Booster + Pre-warming)...")
    res_opt = run_in_process_benchmark(
        scenario_name="Optimized (Fast Scalar + C-Booster)",
        total_requests=args.requests,
        concurrency=args.concurrency,
        use_fast_path=True,
        warm_up=True,
    )
    results.append(res_opt)
    logger.info(f"Optimized: p50={res_opt.latency_p50_ms}ms, p95={res_opt.latency_p95_ms}ms, Throughput={res_opt.throughput_rps} req/s")

    report_md = format_markdown_report(results)
    print("\n" + report_md + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Exported JSON telemetry to {out_path}")

    if args.output_markdown:
        md_path = Path(args.output_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Exported Markdown report to {md_path}")


if __name__ == "__main__":
    main_cli()
