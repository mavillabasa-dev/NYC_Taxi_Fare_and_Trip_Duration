# T-113 — API Benchmarking and Latency Optimisation Report

Load-testing methodology, stated latency budget, component profiling, and before/after
measurements for ticket **T-113**.

All figures come from one run of `scripts/benchmark_api.py` against the current
`models/model.pkl`. The machine-readable source is `docs/benchmark_results.json`.

---

## 1. Stated budget — and what each half is measured against

| Metric | Budget | Measured at | Purpose |
|---|---:|---|---|
| Cold start (first request) | ≤ 50 ms | 1 client | First request after startup must not spike. |
| p50 latency | ≤ 5 ms | 1 client | Dashboard interactions feel immediate. |
| p95 latency | ≤ 15 ms | 1 client | Tight interactive tolerance for almost all requests. |
| p99 latency | ≤ 30 ms | 1 client | Bounded tail. |
| Throughput | ≥ 200 req/s | saturated | Sustained capacity under concurrent load. |

**The "measured at" column is the part that matters, and it was previously missing.**

Latency and throughput are different quantities and cannot be read off the same run.
Percentiles taken while the service is saturated do not measure how long a request takes
to serve — they measure how long it queued waiting for a worker. On this hardware the
same code reports:

| Concurrency | p50 | p95 | p99 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 1.72 ms | 2.15 ms | 2.42 ms | 563 req/s |
| 4 | 5.96 ms | 18.06 ms | — | 560 req/s |
| 10 | 14.95 ms | 26.36 ms | 31.20 ms | 624 req/s |

Throughput barely moves across the three: ~560–620 req/s is the machine's ceiling.
Concurrency beyond that does not produce more work, it produces a queue — and the queue
shows up as latency. **An 8.7× worse p50 at ten clients reflects zero change in the
model, the features, or the code.**

The budget above is a per-request budget, so it is evaluated at concurrency 1
(`LATENCY_CONCURRENCY` in `scripts/benchmark_api.py`). Saturated scenarios report
`— saturated` for latency instead of a pass or fail, because neither verdict would mean
anything.

> An earlier version of this report compared a saturated p50 against the per-request
> budget, concluded the SLA was exceeded, and then published a table marked ✅ PASSED
> using a second, laxer set of thresholds that existed only in prose. Both halves of
> that were wrong. The budget in the table above is the one in the code, and it is the
> only one.

---

## 2. Results

| Scenario | Concurrency | Requests | Cold start | p50 | p90 | p95 | p99 | Max | Throughput | Latency SLA | Throughput SLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Latency (single client)** | 1 | 250 | 8.9 ms | **1.72 ms** | 2.01 ms | 2.15 ms | 2.42 ms | 2.61 ms | 562.7 req/s | ✅ PASSED | ✅ PASSED |
| Baseline (DataFrame + cold) | 10 | 1000 | 5.9 ms | 15.30 ms | 20.87 ms | 28.72 ms | 40.40 ms | 87.76 ms | 581.8 req/s | — saturated | ✅ PASSED |
| Optimised (fast scalar + booster) | 10 | 1000 | 5.7 ms | 14.95 ms | 18.80 ms | 26.36 ms | 31.20 ms | 34.84 ms | 623.8 req/s | — saturated | ✅ PASSED |

**Verdict: the service meets both budgets.** p50 of 1.72 ms against a 5 ms target, p99 of
2.42 ms against 30 ms, and sustained throughput roughly 3× the 200 req/s floor.

---

## 3. Component breakdown

Where a single request's time goes, profiled directly rather than inferred:

| Stage | Duration | Notes |
|---|---:|---|
| Pydantic validation | 0.006 ms | Schema validation, type coercion, ISO timestamp parsing. |
| Fast scalar feature transform | 0.298 ms | Centroid distances, cyclical features, target encoding. |
| LightGBM inference (two models) | 3.150 ms | The dominant cost, by an order of magnitude. |
| *(Legacy DataFrame transform)* | 7.587 ms | The path the fast path replaced. |

Two things follow.

**Inference dominates.** Feature computation is under 10 % of the request. Further
optimisation of the transform would be effort spent on the wrong 3 %.

**The components sum to ~3.5 ms while the single-client p50 is 1.72 ms.** They are not
the same measurement: the breakdown times each stage in isolation, including a cold call
that pays one-off setup, while the p50 comes from warmed steady-state requests. Read the
breakdown for *proportions*, not as an additive budget.

---

## 4. Optimisation: fast scalar path

`SelfContainedTaxiModel.predict_fast` bypasses DataFrame construction and builds the
29-feature vector with scalar arithmetic, then calls the booster directly.

| | Baseline | Optimised | Change |
|---|---:|---:|---:|
| Feature transform | 7.587 ms | 0.298 ms | **−96 %** |
| Throughput (10 clients) | 581.8 req/s | 623.8 req/s | +7.2 % |
| p50 (10 clients) | 15.30 ms | 14.95 ms | −2.3 % |

The transform itself is 25× faster. **End-to-end throughput improves only 7 %**, because
at that point inference and harness overhead dominate — exactly what section 3 predicts.
This is worth stating plainly: the optimisation did what it was designed to do, and the
headline number is still small, because the bottleneck was elsewhere.

`warm_up()` runs one dummy prediction at startup so the first real request does not pay
booster initialisation.

### The fast path is not free

It is a second implementation of the feature computation, and it must stay numerically
identical to `transform_features`. It did not, once: zones without a shapefile centroid
were defaulted to `(0.0, 0.0)` and produced a ~5,400-mile trip. `tests/test_predictor_parity.py`
now compares both paths feature-by-feature for every `LocationID` the API schema accepts.

`ModelService.predict` falls back to the DataFrame path if `predict_fast` raises, and
**logs a warning when it does**. Reaching that branch means the two paths disagree, which
is a bug rather than a tuning knob.

---

## 5. Reproducing

```bash
python scripts/benchmark_api.py --requests 1000 --concurrency 10 --compare \
  --output-json docs/benchmark_results.json
```

The single-client latency scenario always runs first; `--compare` adds the baseline
DataFrame scenario alongside the optimised one. `--concurrency` affects only the
saturated runs.

Requires `models/model.pkl`. Numbers above measured on a 6-core Ryzen 5 PRO 5675U with
LightGBM 4.7.0, Python 3.11. **Absolute timings are machine-specific**; the proportions
in section 3 and the concurrency effect in section 1 are the transferable findings.

`tests/test_benchmark.py` asserts the harness produces coherent output and that the two
SLA verdicts are computed on the right runs. It deliberately asserts **no absolute
timing** — an earlier version required `throughput_rps > 100` and turned every loaded
machine into a red suite.

---

## 6. Caveats

**This is an in-process benchmark.** It drives FastAPI through `TestClient` in the same
Python process, so it excludes the network, the ASGI server's own concurrency model, and
container overhead. Under `uvicorn` with multiple workers the throughput ceiling would
differ — the GIL contention visible in the concurrency table is partly an artefact of
this harness. Use `--mode live-http` against a running server for deployment numbers.

**The model changed since the previous measurement.** The current artifact is the tuned
T-107 winner: 700 trees × 127 leaves, against 300 × 63 before. Single-row inference cost
barely moved (3.15 ms vs 2.20 ms) because per-call overhead dominates tree traversal at
batch size 1 — a bigger model is nearly free here, which would not hold for batch
scoring.

**There is no batch endpoint**, and the dashboard's choropleth issues ~263 sequential
requests to colour the map. That workload is dominated by per-request overhead, not by
the 3 ms of inference this report optimised. A `POST /predict/batch` would help it far
more than anything measured here.
