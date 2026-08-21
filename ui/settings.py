"""Environment configuration for the dashboard."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
	api_url: str = os.getenv("API_URL", "http://localhost:8000")
	dataset_dir: str = os.getenv("DATASET_DIR", "../dataset")


settings = Settings()
