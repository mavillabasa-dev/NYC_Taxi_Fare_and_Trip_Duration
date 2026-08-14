# T-107 — Gradient-boosting ensembles

This experiment trains independent LightGBM and XGBoost regressors for
`fare_amount` and `duration_minutes`. It uses the temporal train/test split produced
by T-104 and the serializable feature transformer produced by T-105.

## Reproducible search

- Method: `RandomizedSearchCV`
- Validation: three forward-only `TimeSeriesSplit` folds from the training period
- Selection metric: MAE
- Default budget: 12 sampled configurations per model/target (48 configurations,
  144 temporal-fold fits, plus four final refits)
- Seed: `src.config.RANDOM_SEED`
- Libraries: LightGBM 4.7.0 and XGBoost 3.2.0
- Parallelism: search candidates use all cores by default; each estimator uses one
  thread to avoid nested oversubscription

The exact search spaces are constants in `src/gradient_boosting.py`. Preprocessing
is cloned into each model pipeline and fitted within each CV fold, never on the test
period.

## Run

### Environment

The repository pins Python 3.11 in `.python-version`. With `uv` installed:

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-dev.txt
source .venv/bin/activate
```

### Data, features, and training

After T-116 and T-104 have produced the cleaned temporal splits, generate the T-105
artifact through its canonical module import and run the experiment:

```bash
python -m src.data_utils
python -m src.preprocessing

python -c "from src.features import build_and_save_feature_pipeline; from src.config import TRAIN_CLEANED_PATH; build_and_save_feature_pipeline(TRAIN_CLEANED_PATH)"

python -m src.gradient_boosting \
  --train dataset/train_cleaned.parquet \
  --test dataset/test_cleaned.parquet \
  --transformer models/feature_pipeline.pkl \
  --output-dir models/t107
```

For a quick smoke run, add `--n-iter 1 --n-jobs 1`. The output directory contains
four candidate pipelines and `t107_results.json`, including test MAE, RMSE, MAPE,
R², training time, warm single-row inference time, best parameters, normalized
feature importance, and any feature whose importance crosses the leakage-review
threshold.

The M2 Pro runs these libraries on CPU (their GPU paths require CUDA and do not use
Apple Metal). To balance throughput and memory during the full search, use two search
workers on a 16 GB Mac or four on a Mac with at least 32 GB:

```bash
# 16 GB M2 Pro
python -m src.gradient_boosting \
  --transformer models/feature_pipeline.pkl \
  --output-dir models/t107 --n-jobs 2
```

## Leakage audit

Every run rejects the targets and all banned post-trip fields before fitting. A
feature contributing at least 65% of total importance is flagged for manual review.
`trip_distance` is allowed by the project contract but remains an optimistic proxy:
production requests must supply a routing estimate, not completed metered distance.

## Current limitation

The raw data, cleaned splits, fitted T-105 pipeline, and T-107 candidate models are
local artifacts under the gitignored `dataset/` and `models/` directories. Generate
them with the repository scripts before running this experiment. The metrics from
the completed run are recorded below. The generated
`models/t107/t107_results.json` is T-109's machine-readable comparison-table input.

## Results

The full-data run completed successfully for all four model/target combinations.
Metrics are measured on the held-out temporal test split and are reported in the
original target units.

| Model | Target | MAE | RMSE | MAPE | R² | Train time | Inference / row |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | Fare amount | **1.264** | **2.960** | **10.48%** | **0.9524** | 1,603.6 s | 5.65 ms |
| XGBoost | Fare amount | 1.266 | 3.009 | 10.56% | 0.9508 | 3,757.5 s | 6.23 ms |
| LightGBM | Duration | **3.365 min** | **5.665 min** | **24.91%** | **0.8218** | 1,577.4 s | 5.65 ms |
| XGBoost | Duration | 3.412 min | 5.733 min | 25.33% | 0.8175 | 3,333.7 s | 6.59 ms |

LightGBM produced the best test metrics and lower training and inference times for
both targets. Final model selection and construction of the API-compatible
`models/model.pkl` bundle remain the responsibility of T-109.

### Feature-importance leakage review

No feature exceeded the 65% dominance threshold in any run. The largest normalized
importance was 43.02% for the LightGBM duration model, so the automated review did
not flag a suspiciously dominant feature. The full normalized importance vectors
and best hyperparameters are retained in `models/t107/t107_results.json`.

The candidate `.joblib` files and JSON report are reproducible local artifacts and
remain gitignored. They are not the final serving artifact; T-109 will select and
package the winning models for the API.
