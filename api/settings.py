"""Configuración de entorno para la API."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
	model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")
	api_host: str = os.getenv("API_HOST", "0.0.0.0")
	api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
