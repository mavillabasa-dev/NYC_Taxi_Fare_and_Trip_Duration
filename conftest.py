"""Configuración compartida para recolectar los dos árboles de tests."""

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))