"""Checkpoint recovery and merge guards for the long T-107 search."""

from argparse import Namespace
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from scripts import checkpoint_boosting as checkpoint
from src.config import TARGET_COLUMNS
from src.gradient_boosting import MODEL_FAMILIES


@pytest.fixture
def run_args(tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    transformer_path = tmp_path / "pipeline.pkl"
    frame = pd.DataFrame({"fare_amount": [10.0], "duration_minutes": [5.0]})
    frame.to_parquet(train_path)
    frame.to_parquet(test_path)
    joblib.dump(None, transformer_path)
    return Namespace(
        run_dir=str(tmp_path / "run"),
        train=str(train_path),
        test=str(test_path),
        transformer=str(transformer_path),
        family="lightgbm",
        target="fare_amount",
        n_iter=12,
        cv_splits=3,
        n_jobs=4,
    )


def fake_experiment(_train, _test, **kwargs):
    family = kwargs["model_families"][0]
    target = kwargs["targets"][0]
    return [
        SimpleNamespace(
            model_family=family,
            target=target,
            estimator=DummyRegressor(),
            report_dict=lambda: {
                "model_family": family,
                "target": target,
                "best_params": {"model__n_estimators": 200},
                "cv_mae": 2.0,
                "metrics": {"mae": 2.5},
                "training_seconds": 3.0,
            },
        )
    ]


def test_completed_pairs_resume_and_merge_only_when_all_inputs_match(run_args, monkeypatch):
    monkeypatch.setattr(checkpoint, "run_experiments", fake_experiment)
    run_dir = checkpoint.Path(run_args.run_dir)

    checkpoint.run_pair(run_args)
    first = checkpoint.checkpoint_dir(run_dir, run_args.family, run_args.target)
    first_hash = checkpoint.sha256_file(first / "t107_results.json")
    checkpoint.run_pair(run_args)
    assert checkpoint.sha256_file(first / "t107_results.json") == first_hash

    with pytest.raises(FileNotFoundError):
        checkpoint.merge(run_args)

    for family in MODEL_FAMILIES:
        for target in TARGET_COLUMNS:
            run_args.family = family
            run_args.target = target
            checkpoint.run_pair(run_args)

    checkpoint.merge(run_args)
    combined = checkpoint.json.loads((run_dir / "t107_results.json").read_text())
    assert len(combined["runs"]) == 4
    checkpoint.merge(run_args)

    pd.DataFrame({"fare_amount": [11.0], "duration_minutes": [5.0]}).to_parquet(run_args.train)
    with pytest.raises(ValueError, match="Input changed"):
        checkpoint.merge(run_args)
    with pytest.raises(ValueError, match="settings or inputs differ"):
        checkpoint.run_pair(run_args)


def test_failed_search_does_not_commit_checkpoint(run_args, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("interrupted search")

    monkeypatch.setattr(checkpoint, "run_experiments", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        checkpoint.run_pair(run_args)
    assert not checkpoint.checkpoint_dir(
        checkpoint.Path(run_args.run_dir), run_args.family, run_args.target
    ).exists()


def test_merge_rejects_mixed_search_settings(run_args, monkeypatch):
    monkeypatch.setattr(checkpoint, "run_experiments", fake_experiment)
    checkpoint.run_pair(run_args)
    run_args.n_jobs = 2
    for family in MODEL_FAMILIES:
        for target in TARGET_COLUMNS:
            if (family, target) == ("lightgbm", "fare_amount"):
                continue
            run_args.family = family
            run_args.target = target
            checkpoint.run_pair(run_args)

    with pytest.raises(ValueError, match="different data, source, library versions, or settings"):
        checkpoint.merge(run_args)
