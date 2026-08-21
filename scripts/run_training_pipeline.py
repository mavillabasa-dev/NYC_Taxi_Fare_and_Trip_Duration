"""scripts/run_training_pipeline.py — End-to-End ML Training Pipeline Runner.

This script orchestrates the full offline training pipeline from raw data acquisition to production artifact export:
1. Ingestion & Zone Centroid Derivation (src.data_utils)
2. Preprocessing & Temporal Train/Test Splits (src.preprocessing)
3. Feature Engineering Pipeline Fitting (src.features)
4. Model Selection, Residual Analysis & Production Packaging (src.model_selection)
5. Model Isolation & Unpickling Acceptance Verification (tests.verify_isolation)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline_runner")


def run_pipeline() -> None:
    t_start = time.time()
    logger.info("=================================================================")
    logger.info("  STARTING END-TO-END NYC TAXI MACHINE LEARNING TRAINING PIPELINE")
    logger.info("=================================================================")

    # Stage 1: Ingestion & Spatial Centroids (T-116)
    logger.info("[1/5] Running Data Ingestion & Centroid Extraction...")
    from src.data_utils import ingest_all_data

    ingest_all_data()

    # Stage 2: Data Cleaning & Temporal Splitting (T-104)
    logger.info("[2/5] Running Data Cleaning & Preprocessing...")
    from src.preprocessing import clean_and_preprocess_dataset

    clean_and_preprocess_dataset()

    # Stage 3: Feature Engineering Pipeline (T-105)
    logger.info("[3/5] Fitting Feature Engineering Pipeline...")
    from src.config import TRAIN_CLEANED_PATH
    from src.features import build_and_save_feature_pipeline

    build_and_save_feature_pipeline(TRAIN_CLEANED_PATH)

    # Stage 4: Model Selection, Benchmarking & Production Export (T-109)
    logger.info("[4/5] Training Winning Models & Exporting Production Artifact...")
    from src.model_selection import run_full_model_selection_pipeline

    run_full_model_selection_pipeline()

    # Stage 5: Artifact Isolation Acceptance Test (T-109 / T-114)
    logger.info("[5/5] Running Artifact Isolation Acceptance Verification...")
    from src.config import MODEL_PATH
    from src.model_selection import verify_model_isolation

    isolation_passed = verify_model_isolation(MODEL_PATH)
    if not isolation_passed:
        raise RuntimeError("Artifact isolation acceptance verification failed!")

    total_time = time.time() - t_start
    logger.info("=================================================================")
    logger.info(f"  PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_pipeline()
