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
them with the repository scripts before running this experiment. The metrics from a
completed run are recorded below and the generated `models/t107/t107_results.json`
is T-109's machine-readable comparison-table input.

## Results

Pending the reproducible full-data run.
