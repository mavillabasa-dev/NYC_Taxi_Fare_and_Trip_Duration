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
import sys
import time

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
    from src.data_utils import main as run_ingestion

    run_ingestion()

    # Stage 2: Data Cleaning & Temporal Splitting (T-104)
    logger.info("[2/5] Running Data Cleaning & Preprocessing...")
    from src.preprocessing import run_preprocessing_pipeline

    run_preprocessing_pipeline()

    # Stage 3: Feature Engineering Pipeline (T-105)
    logger.info("[3/5] Fitting Feature Engineering Pipeline...")
    from src.features import run_feature_engineering_pipeline

    run_feature_engineering_pipeline()

    # Stage 4: Model Selection, Benchmarking & Production Export (T-109)
    logger.info("[4/5] Training Winning Models & Exporting Production Artifact...")
    from src.model_selection import run_full_model_selection_pipeline

    run_full_model_selection_pipeline()

    # Stage 5: Artifact Isolation Acceptance Test (T-109 / T-114)
    logger.info("[5/5] Running Artifact Isolation Acceptance Verification...")
    from tests.verify_isolation import verify_artifact_isolation
    import os
    from src.config import MODEL_PATH

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_dir = os.path.join(repo_root, "api")
    verify_artifact_isolation(MODEL_PATH, api_dir)

    total_time = time.time() - t_start
    logger.info("=================================================================")
    logger.info(f"  PIPELINE COMPLETE SUCCESSFULLY IN {total_time:.2f}s")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_pipeline()
