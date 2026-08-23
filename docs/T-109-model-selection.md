# T-109 — Model Selection, Benchmarking, and Evaluation Report

Formal model selection, benchmark comparison, feature importance analysis, residual
diagnostics, and the self-contained artifact specification for ticket **T-109**.

Every number in this document was regenerated from a single end-to-end run and is
reproducible with the commands in section 7. The machine-readable source is
`models/model_comparison.json`, written by `src/model_selection.py` on every run — if
a figure here disagrees with that file, the file is right.

---

## 1. Model benchmark leaderboard

Trained on `train_cleaned.parquet` (pickup before 2022-05-23, 2,402,868 rows) and
evaluated on `test_cleaned.parquet` (2022-05-23 onward, 899,623 rows). Metrics are in
original units — dollars for fare, minutes for duration.

| Model | Target | MAE | RMSE | MAPE | R² | Train time | Latency / row |
|---|---|---:|---:|---:|---:|---:|---:|
| Trivial mean baseline | fare | 8.8515 | 13.5709 | 77.69 % | −0.0002 | 0.01 s | 0.02 ms |
| Trivial mean baseline | duration | 9.2712 | 13.4220 | 99.09 % | −0.0002 | 0.01 s | 0.03 ms |
| Decision tree | fare | 1.4639 | 2.8936 | 12.28 % | 0.9545 | 25.5 s | 0.55 ms |
| Decision tree | duration | 3.9859 | 6.5208 | 30.38 % | 0.7639 | 23.7 s | 0.54 ms |
| MLP (64, 32) | fare | 2.1542 | 3.4451 | 19.89 % | 0.9355 | 74.8 s | 0.91 ms |
| MLP (64, 32) | duration | 6.1504 | 8.8628 | 51.68 % | 0.5639 | 80.0 s | 1.31 ms |
| XGBoost (tuned) | fare | 1.2726 | 3.0334 | 10.54 % | 0.9500 | — | — |
| XGBoost (tuned) | duration | 3.3831 | 5.6854 | 25.12 % | 0.8205 | — | — |
| **LightGBM (tuned, winner)** | **fare** | **1.2639** | **2.9600** | **10.48 %** | **0.9524** | — | — |
| **LightGBM (tuned, winner)** | **duration** | **3.3650** | **5.6653** | **24.91 %** | **0.8218** | — | — |
| LightGBM as shipped | fare | 1.2701 | 2.9789 | 10.53 % | 0.9518 | 76.2 s | 1.07 ms |
| LightGBM as shipped | duration | 3.4087 | 5.7051 | 25.40 % | 0.8193 | 75.9 s | 1.07 ms |

### Three caveats about reading this table

**The metrics are not measured identically across families.** `src/train.py` (baseline,
tree) and `src/mlp.py` compute MAPE by masking zero denominators; `src/gradient_boosting.py`
uses scikit-learn's `mean_absolute_percentage_error`, which clamps instead. On cleaned
data —where fare ≥ 2.50 and duration ≥ 0.5— the two agree, but the MAPE column mixes two
definitions. Latency is worse: baseline, tree and MLP average 1,000 repetitions, the
gradient-boosting module averages 30, and the two rows differ in whether the feature
transformer is inside the timed block. **Treat the latency column as indicative only.**

**The gradient-boosting rows come from the T-107 search, not from a refit here.** Their
train time is the whole `RandomizedSearchCV` (~20 min for LightGBM, ~80 min for XGBoost),
which is not comparable to a single fit — hence the dashes.

**The last two rows are the model that actually ships.** They differ from the winner rows
by 0.5 % on fare and 1.3 % on duration. That gap is explained in section 3 and is not a
defect, but it does mean the deployed model is measurably —if marginally— worse than the
row that won the comparison.

---

## 2. Winning model justification

**Selected: LightGBM (`LGBMRegressor`) for both targets, as two independent single-output
models.**

**Accuracy.** Lowest MAE on both targets. Against the trivial baseline that is an 85.7 %
reduction in fare error and 63.7 % in duration error — the margin that justifies training
a model at all. Against the decision tree, 13.6 % and 15.6 %. Against the MLP, 41.3 % and
45.3 %.

**The MLP result is worth stating plainly:** it lost decisively, especially on duration
(R² 0.5639 against 0.8218). This is the expected outcome for tabular data and is recorded
here so the option is closed rather than left open.

**LightGBM over XGBoost** is the narrower call. Accuracy differs by 0.7 % on fare and
0.5 % on duration — within noise for a practical decision. The real separator is cost:
XGBoost took roughly **4× longer to search** for that marginal difference, and its
per-row inference was slower in every measurement. Given equivalent accuracy, the cheaper
model wins.

**Latency.** The shipped model answers in ~1.07 ms per row measured on the bare booster.
The end-to-end request budget is T-113's concern, and it must be re-measured: the tuned
winner is 700 trees × 127 leaves, against 300 × 63 in the previously deployed
configuration. Serving latency will rise.

### Loss function

Both models are trained with `objective="regression_l1"`, matching the metric T-107
selected on. This is a product decision as much as a technical one:

- Fares are heavily right-skewed (airport flat rates, long trips). L1 predicts the
  conditional median and is robust to that tail.
- The user-facing number is a quote. **MAE and MAPE are the user experience**; RMSE is not.
- On duration, L1 wins on all four metrics. On fare it trades ~0.7 points of R² for ~9 %
  better MAE.

An earlier version of this pipeline trained with LightGBM's default L2 objective while
reporting the L1 numbers from T-107 — a mismatch that shipped a model with 12.5 % worse
MAE than the figure published to justify it. `LGBM_BASE_KWARGS` in
`src/model_selection.py` now pins the objective so it cannot drift from the selection
criterion again.

---

## 3. Why the shipped model differs from the winner row

`src/model_selection.py` reads `models/gbm/t107_results.json` and refits with the
hyperparameters the search selected, so the two should agree. They agree to 0.5 % on
fare. The residual difference has one identified cause:

**T-107 and T-109 build different features for the duration model.** T-107 fits the
transformer inside its pipeline against the single target being tuned, so
`PULocationID_target_enc` encodes *mean duration per zone* for the duration run. T-109
uses one shared `feature_pipeline.pkl`, fitted on both targets, whose encoder takes the
first column — *mean fare per zone* — for both models.

This is a design consequence, not a bug: the production artifact carries a single
`target_encodings` dictionary shared by both estimators. Splitting it per target would
improve duration accuracy at the cost of a larger artifact and a changed bundle
structure. **Open decision, deliberately not taken here.** It explains why duration
diverges more (1.3 %) than fare (0.5 %).

### Winning hyperparameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 700 |
| `num_leaves` | 127 |
| `max_depth` | 12 |
| `learning_rate` | 0.05 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 |

`subsample` requires `subsample_freq > 0` in LightGBM or it is silently ignored;
`LGBM_BASE_KWARGS` sets it, matching `src/gradient_boosting._make_regressor`.

---

## 4. Feature importance

Split gain, normalized, from the shipped model. Full vectors in
`models/model_comparison.json`.

| Rank | Fare | gain | Duration | gain |
|---:|---|---:|---|---:|
| 1 | `manhattan_distance` | 32.11 % | `trip_distance` | 43.02 % |
| 2 | `trip_distance` | 26.04 % | `haversine_distance` | 19.70 % |
| 3 | `haversine_distance` | 20.82 % | `cos_hour` | 4.26 % |
| 4 | `haversine_ratio` | 5.74 % | `haversine_ratio` | 4.20 % |
| 5 | `RatecodeID` | 3.65 % | `do_lat` | 3.76 % |
| 6 | `do_lon` | 1.38 % | `pickup_hour` | 3.71 % |

**Leakage audit: passed.** No feature reaches the 65 % review threshold; the highest is
`trip_distance` at 43.02 % on duration.

Two observations that matter more than the ranking:

**Distance dominates in aggregate** — roughly 85 % of fare gain and 67 % of duration gain
come from the four distance features combined. That is correct for a metered taxi, and it
also means the `trip_distance` proxy assumption carries most of the model's weight. Any
bias in how that value is supplied at prediction time propagates almost undiluted. The
dashboard currently supplies a straight-line haversine estimate, which systematically
underestimates road distance.

**Duration leans on time, fare does not.** `cos_hour` and `pickup_hour` together reach
~8 % for duration and ~2 % for fare. The model has learned what the metrics already
suggest: fare is close to deterministic given distance, while duration depends on traffic.

---

## 5. Residual diagnostics

Residuals are `y_true − y_pred`, so **a negative bias means the model overestimates**.
Computed on the 899,623-trip test split.

### By pickup borough

| Borough | Trips | Fare MAE | Fare bias | Duration MAE | Duration bias |
|---|---:|---:|---:|---:|---:|
| Manhattan | 801,257 | $1.187 | −0.308 | 2.94 min | −0.98 |
| Queens | 84,038 | $1.903 | −0.274 | **7.57 min** | **−3.74** |
| Unknown | 9,876 | $2.254 | −0.259 | 5.28 min | −1.80 |
| Brooklyn | 4,070 | $1.894 | +0.204 | 4.22 min | −0.12 |
| Bronx | 357 | $3.590 | +1.776 | 4.75 min | +0.63 |
| EWR | 21 | $11.444 | −2.764 | 6.00 min | −1.31 |
| Staten Island | 4 | $17.208 | −7.938 | 13.48 min | −3.90 |

**89 % of the test set is Manhattan pickups**, and that is where the model is strongest.
The headline metrics are effectively Manhattan metrics.

**Queens is the actionable finding.** Duration MAE is 2.6× the Manhattan figure, with a
−3.74 min bias — the model **systematically under-predicts** how long a Queens pickup
takes. These are largely JFK and LaGuardia runs, where highway traffic dominates and the
model has no traffic signal.

Staten Island (4 trips) and EWR (21 trips) carry no statistical weight. They are listed
for completeness; **do not quote their MAE.**

### By trip distance

| Bucket | Trips | Fare MAE | Duration MAE |
|---|---:|---:|---:|
| Short (< 2 mi) | 460,286 | $0.945 | 2.11 min |
| Medium (2–10 mi) | 354,259 | $1.523 | 3.85 min |
| Long (> 10 mi) | 85,078 | $1.975 | 8.60 min |

Error grows with distance for both targets, but far faster for duration (4.1× from short
to long) than for fare (2.1×). Longer trips accumulate more traffic uncertainty while
fare stays anchored to the meter.

### By rate code

| Rate code | Trips | Fare MAE | Fare bias | Duration MAE |
|---|---:|---:|---:|---:|
| 1 · Standard | 852,688 | $1.228 | −0.346 | 3.07 min |
| 2 · JFK flat | 39,619 | **$0.115** | −0.008 | **9.64 min** |
| 3 · Newark | 3,055 | $4.082 | +0.879 | 8.87 min |
| 4 · Nassau/Westchester | 1,292 | $17.423 | +2.356 | 9.99 min |
| 5 · Negotiated | 2,963 | $18.887 | +6.352 | 9.40 min |
| 6 · Group ride | 6 | $60.961 | −60.961 | 4.25 min |

**The JFK flat rate is the sharpest result in the report.** Fare MAE of 11 cents across
39,619 trips: the model learned the fixed tariff exactly. The same trips have the worst
duration error of any high-volume slice (9.64 min) — the price is fixed, the journey is
not.

**Rate codes 4 and 5 are near-unpredictable by construction.** Negotiated fares are not
formula-driven, and out-of-city trips vary enormously. Both are also over-predicted
(positive bias). Together they are 0.5 % of trips but contribute disproportionately to
RMSE — which is most of why RMSE (2.98) sits at 2.3× MAE (1.27).

Rate code 6 has six trips. It is noise.

### By hour of day

| | Best | Worst |
|---|---|---|
| Duration MAE | 02 h · 1.78 min · 03 h · 1.84 · 01 h · 1.92 | 15 h · 4.69 min · 16 h · 4.53 · 14 h · 4.52 |
| Fare MAE | 03 h · $0.767 · 06 h · $0.783 · 02 h · $0.792 | 15 h · $1.704 · 14 h · $1.611 · 16 h · $1.607 |

Empty streets between 1 and 3 a.m. are the most predictable hours by a wide margin.

Note that **the worst hours are mid-afternoon (14–16 h), not the evening rush.** The
error peaks before the classic 17–19 h window, which is the kind of thing worth knowing
before writing a caption about rush-hour congestion.

---

## 6. Production artifact contract (`models/model.pkl`)

```python
{
    "model": <SelfContainedTaxiModel>,
    "feature_order": [
        "PULocationID", "DOLocationID", "tpep_pickup_datetime",
        "passenger_count", "RatecodeID", "trip_distance",
    ],
    "version": "1.0.0-lightgbm",
}
```

**Zero dependency on `src/`.** `SelfContainedTaxiModel` lives in
`api/app/model/predictor.py` and is imported as `app.model.predictor` — the only spelling
that resolves inside the container, where `COPY . .` from `./api` flattens the tree.
Pickle records a class's module path, so importing it any other way produces a bundle that
fails to load in production. `tests/verify_isolation.py` asserts this in a subprocess with
the repo root stripped from `sys.path`.

**Embedded lookups.** Zone centroids for all 265 `LocationID`s and the training-fitted
target encodings travel inside the object. Nothing is read from disk at inference time,
so there is no way for serving to drift from training on these values.

**Predict contract.** `model.predict(raw_df)` returns `np.ndarray` of shape `(N, 2)`:
column 0 is fare (≥ 2.50), column 1 is duration in minutes (≥ 0.5). `model.predict_fast(payload)`
is a scalar fast path for single rows, numerically identical to `predict` — enforced by
`tests/test_predictor_parity.py` across every `LocationID` the API schema accepts.

> `LocationID` 264 and 265 have no shapefile geometry and reach the lookup as `(nan, nan)`.
> Both paths keep the coordinates NaN and zero the distances. Anything else fabricates a
> trip thousands of miles long.

---

## 7. Reproduction

```bash
pip install -r requirements-dev.txt

python -m src.data_utils        # T-116 · download + zone centroids
python -m src.preprocessing     # T-104 · clean + temporal split
python -c "from src.features import build_and_save_feature_pipeline; from src.config import TRAIN_CLEANED_PATH; build_and_save_feature_pipeline(TRAIN_CLEANED_PATH)"

# T-107 search — required before T-109, several hours
python -m src.gradient_boosting --transformer models/feature_pipeline.pkl --output-dir models/gbm

python -m src.train             # T-106 · baseline + decision tree rows
python -m src.mlp               # T-108 · MLP rows
python -m src.model_selection   # T-109 · refit winner, export artifact, write report

pytest
```

`src/model_selection.py` warns loudly if `models/gbm/t107_results.json` is missing and
falls back to untuned hyperparameters. **If you see that warning, the artifact is not the
tuned winner** and the numbers in this document will not reproduce.

Steps 2–4 and the shortcut `python -m scripts.run_training_pipeline` run everything except
the T-107 search, in about 75 seconds.

### Environment for the figures above

LightGBM 4.7.0 · pandas 2.2.3 · numpy 2.2.1 · Python 3.11, on a 6-core Ryzen 5 PRO 5675U.
Metrics reproduce across machines; timings do not.

---

## 8. Container dependencies (T-114)

Serving `models/model.pkl` requires, pinned exactly in `api/requirements.txt`:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.2.1
lightgbm==4.7.0
```

The last four also appear in the root `requirements.txt` and **must match exactly**: the
pickle is the only interface between the training host and the container, and a model
serialized under one numpy/pandas and unpickled under another may fail to load or behave
differently. `requirements-dev.txt` installs both files at once, so pip reports a conflict
if they ever drift apart.
