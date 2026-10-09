"""Run T-107 one model/target pair at a time with verified checkpoints.

The original T-107 CLI saves only after all four searches finish. This wrapper
uses the same search implementation and settings, but commits each pair only
after its estimator and report have been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMNS,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
)
from src.gradient_boosting import MODEL_FAMILIES, SearchConfig, run_experiments, save_results

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    REPO_ROOT / "src" / "config.py",
    REPO_ROOT / "src" / "features.py",
    REPO_ROOT / "src" / "gradient_boosting.py",
    REPO_ROOT / "requirements.txt",
    Path(__file__).resolve(),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recorded_path(path: Path) -> str:
    """Keep repository inputs portable between host and trainer-container paths."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_recorded_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "train": Path(args.train).resolve(),
        "test": Path(args.test).resolve(),
        "transformer": Path(args.transformer).resolve(),
    }
    files = {name: recorded_path(path) for name, path in input_paths.items()}
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    hashes.update({str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in SOURCE_FILES})
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "lightgbm": __import__("lightgbm").__version__,
        "xgboost": __import__("xgboost").__version__,
        "scikit_learn": __import__("sklearn").__version__,
    }
    settings = {
        "n_iter": args.n_iter,
        "cv_splits": args.cv_splits,
        "n_jobs": args.n_jobs,
        "scoring": SearchConfig().scoring,
        "random_seed": RANDOM_SEED,
    }
    identity = {"paths": files, "hashes": hashes, "versions": versions, "settings": settings}
    identity["fingerprint"] = fingerprint(
        {
            "hashes": hashes,
            "versions": versions,
            "settings": settings,
        }
    )
    return identity


def checkpoint_dir(run_dir: Path, family: str, target: str) -> Path:
    return run_dir / "checkpoints" / f"{family}_{target}"


def check_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    report_path = path / "t107_results.json"
    model_path = path / f"t107_{manifest['family']}_{manifest['target']}.joblib"
    if sha256_file(report_path) != manifest["report_sha256"]:
        raise ValueError(f"Checkpoint report changed: {report_path}")
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError(f"Checkpoint model changed: {model_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs", [])
    if len(runs) != 1 or (runs[0]["model_family"], runs[0]["target"]) != (
        manifest["family"],
        manifest["target"],
    ):
        raise ValueError(f"Checkpoint contains the wrong pair: {path}")
    return manifest, report


def run_pair(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    destination = checkpoint_dir(run_dir, args.family, args.target)
    identity = run_identity(args)
    if destination.exists():
        manifest, _ = check_checkpoint(destination)
        if manifest["fingerprint"] != identity["fingerprint"]:
            raise ValueError(f"Checkpoint settings or inputs differ: {destination}")
        print(f"Already complete: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    transformer = joblib.load(args.transformer)
    config = SearchConfig(
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    runs = run_experiments(
        train_df,
        test_df,
        transformer=transformer,
        targets=(args.target,),
        model_families=(args.family,),
        config=config,
    )
    if len(runs) != 1:
        raise ValueError("Expected exactly one search result")

    # A failed/interrupted search leaves no valid checkpoint. An interrupted
    # process may leave a .partial-* directory, which the next run ignores.
    with tempfile.TemporaryDirectory(prefix=".partial-", dir=destination.parent) as temp:
        temporary = Path(temp)
        save_results(runs, temporary)
        report_path = temporary / "t107_results.json"
        model_path = temporary / f"t107_{args.family}_{args.target}.joblib"
        manifest = {
            **identity,
            "family": args.family,
            "target": args.target,
            "report_sha256": sha256_file(report_path),
            "model_sha256": sha256_file(model_path),
        }
        (temporary / "checkpoint.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(temporary, destination)
    print(f"Saved checkpoint: {destination}")


def merge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifests = []
    reports = []
    for family in MODEL_FAMILIES:
        for target in TARGET_COLUMNS:
            path = checkpoint_dir(run_dir, family, target)
            manifest, report = check_checkpoint(path)
            manifests.append(manifest)
            reports.append(report)

    first = manifests[0]
    if any(item["fingerprint"] != first["fingerprint"] for item in manifests[1:]):
        raise ValueError("Checkpoints used different data, source, library versions, or settings")
    for name, path_text in first["paths"].items():
        if sha256_file(resolve_recorded_path(path_text)) != first["hashes"][name]:
            raise ValueError(f"Input changed after checkpoint creation: {name}")
    for path in SOURCE_FILES:
        if sha256_file(path) != first["hashes"][str(path.relative_to(REPO_ROOT))]:
            raise ValueError(f"Source changed after checkpoint creation: {path}")
    if any(report["library_versions"] != reports[0]["library_versions"] for report in reports[1:]):
        raise ValueError("Checkpoints have different boosting library versions")

    combined = {
        "ticket": "T-107",
        "search_method": reports[0]["search_method"],
        "library_versions": reports[0]["library_versions"],
        "checkpoint_fingerprint": first["fingerprint"],
        "runs": [report["runs"][0] for report in reports],
    }
    output = run_dir / "t107_results.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != combined:
            raise ValueError(f"Existing report differs; preserve it before merging: {output}")
        print(f"Already merged: {output}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=run_dir, prefix=".t107-", suffix=".tmp", delete=False
    ) as handle:
        temporary_name = handle.name
        json.dump(combined, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_name, output)
    print(f"Merged 4 checkpoints: {output}")


def status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    for family in MODEL_FAMILIES:
        for target in TARGET_COLUMNS:
            path = checkpoint_dir(run_dir, family, target)
            if not path.exists():
                print(f"{family}/{target}: pending")
                continue
            manifest, report = check_checkpoint(path)
            run = report["runs"][0]
            print(
                f"{family}/{target}: complete, CV MAE={run['cv_mae']:.4f}, "
                f"fit={run['training_seconds']:.0f}s, fingerprint={manifest['fingerprint'][:12]}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(Path(MODELS_DIR) / "gbm"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run and checkpoint one model/target search")
    run.add_argument("--family", choices=MODEL_FAMILIES, required=True)
    run.add_argument("--target", choices=TARGET_COLUMNS, required=True)
    run.add_argument("--train", default=TRAIN_CLEANED_PATH)
    run.add_argument("--test", default=TEST_CLEANED_PATH)
    run.add_argument("--transformer", default=str(Path(MODELS_DIR) / "feature_pipeline.pkl"))
    run.add_argument("--n-iter", type=int, default=12)
    run.add_argument("--cv-splits", type=int, default=3)
    run.add_argument("--n-jobs", type=int, default=4)
    subparsers.add_parser("merge", help="Validate four checkpoints and make the T-107 report")
    subparsers.add_parser("status", help="Print concise checkpoint status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "run":
            run_pair(args)
        elif args.command == "merge":
            merge(args)
        else:
            status(args)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"Checkpoint error: {exc}") from exc


if __name__ == "__main__":
    main()
