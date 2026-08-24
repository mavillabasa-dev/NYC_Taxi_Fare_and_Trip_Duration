# T-107 — Gradient-boosting ensembles

This experiment trains independent LightGBM and XGBoost regressors for
`fare_amount` and `duration_minutes`. It uses the temporal train/test split produced
by T-104 and the serializable feature transformer produced by T-105.

## Reproducible search

- Method: `RandomizedSearchCV`
- Validation: three forward-only `TimeSeriesSplit` folds from the training period
- Selection metric: MAE (`scoring="neg_mean_absolute_error"`)
- Objective: `regression_l1` for LightGBM, `reg:absoluteerror` for XGBoost — the loss
  matches the selection metric, so the search optimises the thing it is judged on
- Budget: 12 sampled configurations per model/target (48 configurations,
  144 temporal-fold fits, plus four final refits)
- Seed: `src.config.RANDOM_SEED`
- Determinism: LightGBM runs with `deterministic=True`, `force_col_wise=True` and one
  thread per estimator, so results do not drift with core count
- Libraries: LightGBM 4.7.0 and XGBoost 3.2.0

The exact search spaces are constants in `src/gradient_boosting.py`. Preprocessing
is cloned into each model pipeline and fitted within each CV fold, never on the test
period.

## Run

### Environment

The repository pins Python 3.11 in `.python-version`. Dependency versions are pinned
in `requirements.txt` and must stay in sync with `api/requirements.txt` — see the
comment block at the top of either file.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Data, features, and training

After T-116 and T-104 have produced the cleaned temporal splits, generate the T-105
artifact and run the experiment:

```bash
python -m src.data_utils
python -m src.preprocessing

python -c "from src.features import build_and_save_feature_pipeline; from src.config import TRAIN_CLEANED_PATH; build_and_save_feature_pipeline(TRAIN_CLEANED_PATH)"

python -m src.gradient_boosting \
  --transformer models/feature_pipeline.pkl \
  --output-dir models/gbm
```

For a quick smoke run, add `--n-iter 1 --n-jobs 1`. The output directory contains
four candidate pipelines and `t107_results.json`, including test MAE, RMSE, MAPE,
R², training time, warm single-row inference time, best parameters, normalized
feature importance, and any feature whose importance crosses the leakage-review
threshold.

> **Budget the wall clock.** The full search is a multi-hour job. On a 6-core mobile
> Ryzen 5 PRO 5675U it took **3 h 21 min**; XGBoost alone accounted for 69 % of that.
> Use `--n-jobs` to cap the number of parallel search workers if memory is tight.

## Leakage audit

Every run rejects the targets and all banned post-trip fields before fitting. A
feature contributing at least 65 % of total importance is flagged for manual review.
`trip_distance` is allowed by the project contract but remains an optimistic proxy:
production requests must supply a routing estimate, not completed metered distance.

## Results

Full-data run over all four model/target combinations, measured on the held-out
temporal test split (899,623 trips) in original target units.

| Model | Target | MAE | RMSE | MAPE | R² | Search time | Inference / row |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LightGBM** | Fare amount | **1.2639** | **2.9600** | **10.48 %** | **0.9524** | 1,180 s | 9.59 ms |
| XGBoost | Fare amount | 1.2726 | 3.0334 | 10.54 % | 0.9500 | 4,880 s | 12.23 ms |
| **LightGBM** | Duration | **3.3650 min** | **5.6653 min** | **24.91 %** | **0.8218** | 1,132 s | 9.18 ms |
| XGBoost | Duration | 3.3831 min | 5.6854 min | 25.12 % | 0.8205 | 4,702 s | 15.44 ms |

LightGBM produced the best test metrics on both targets and trained roughly **4×
faster** than XGBoost for a marginal accuracy gain — the clearest part of the
decision. The accuracy gap between the two families is small (0.7 % MAE on fare,
0.5 % on duration); the runtime gap is not.

> Search time is the whole `RandomizedSearchCV` for that model/target, not a single
> fit. Inference time is measured through the **full pipeline**, including the feature
> transformer, which is why it is an order of magnitude higher than the figure quoted
> in T-109 — that one measures the bare booster on pre-transformed features. The two
> are not comparable; see the note in `docs/T-109-model-selection.md`.

### Winning hyperparameters

Both targets selected the same configuration, the largest in the search space:

| Parameter | Value |
| --- | ---: |
| `n_estimators` | 700 |
| `num_leaves` | 127 |
| `max_depth` | 12 |
| `learning_rate` | 0.05 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 |

Cross-validated MAE: 1.2208 (fare), 3.0920 (duration).

That the search landed on the ceiling of every capacity parameter (`n_estimators`,
`num_leaves`, `max_depth`) is a signal worth recording: **the space may be too small.**
A wider search could plausibly do better, at the cost of a larger, slower model. The
current winner is already 700 trees × 127 leaves, which more than doubles serving
latency compared with the 300 × 63 configuration previously deployed — a trade-off
T-113 has to re-measure.

### Feature-importance leakage review

No feature exceeded the 65 % dominance threshold in any run. The largest single
normalized importance was **43.02 %**, for `trip_distance` in the LightGBM duration
model, so the automated review did not flag a suspiciously dominant feature.

Distance features nonetheless dominate in aggregate — around 85 % of total gain for
fare and 67 % for duration. That is expected for a metered taxi, but it means the
`trip_distance` proxy assumption carries most of the model's weight, and any bias in
how that value is supplied at serving time propagates almost undiluted into the
prediction.

> **Known gap in the report.** Feature importances are written to `t107_results.json`
> under positional names (`feature_4`, `feature_21`) rather than real column names,
> because column labels are lost passing through the `Pipeline`. The automated
> threshold check still works, but a human reading the file cannot tell which feature
> dominates without mapping indices by hand. `models/model_comparison.json`, written
> by T-109, carries the same importances with proper names.

### Reproducibility

The metrics above were regenerated from scratch and match the previously recorded run
to four decimal places for LightGBM on both targets. XGBoost differs in the third
decimal (1.2726 vs. 1.266 on fare MAE), which is consistent with it not being pinned
to a deterministic mode the way LightGBM is.

Search and training times are machine-dependent and will not reproduce; the metrics
are the part that should.

## Current limitation

The raw data, cleaned splits, fitted T-105 pipeline, and T-107 candidate models are
local artifacts under the gitignored `dataset/` and `models/` directories. Generate
them with the repository scripts before running this experiment.

`models/gbm/t107_results.json` is **T-109's machine-readable input**: since
`src/model_selection.py` reads it to recover the winning hyperparameters, the file
must exist before running T-109, or the exported artifact falls back to untuned values
and logs a warning saying so.
