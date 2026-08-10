"""Configuración de entorno para el dashboard."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
	api_url: str = os.getenv("API_URL", "http://localhost:8000")


settings = Settings()
