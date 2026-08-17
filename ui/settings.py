"""Configuración de entorno para el dashboard."""

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_default_dataset_dir() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "dataset").exists():
        return str(repo_root / "dataset")
    if (Path.cwd() / "dataset").exists():
        return str(Path.cwd() / "dataset")
    if (Path.cwd().parent / "dataset").exists():
        return str(Path.cwd().parent / "dataset")
    return "../dataset"


@dataclass(frozen=True)
class Settings:
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    dataset_dir: str = os.getenv("DATASET_DIR", _resolve_default_dataset_dir())


settings = Settings()
