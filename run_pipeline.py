"""
Predictive Maintenance Pipeline — Main Runner
==============================================
End-to-end pipeline: data ingestion → preprocessing → feature engineering
→ model training → evaluation.

Usage:
    python run_pipeline.py --config config/config.yaml
"""

import argparse
import time

from src.utils import load_config, setup_logger
from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_train, preprocess_test
from src.feature_engineering import engineer_features
from src.train import train_models
from src.evaluate import evaluate_models

logger = setup_logger("pipeline")


def run(config_path: str) -> None:
    """Execute the full predictive maintenance pipeline."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE PIPELINE — STARTING")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    # Step 1: Data Ingestion
    logger.info("-" * 40)
    logger.info("STEP 1: Data Ingestion")
    logger.info("-" * 40)
    train_df, test_df, rul_df = ingest_data(config)

    # Step 2: Preprocessing
    logger.info("-" * 40)
    logger.info("STEP 2: Preprocessing")
    logger.info("-" * 40)
    train_df, scaler = preprocess_train(train_df, config)
    test_df = preprocess_test(test_df, rul_df, config, scaler)

    # Step 3: Feature Engineering
    logger.info("-" * 40)
    logger.info("STEP 3: Feature Engineering")
    logger.info("-" * 40)
    train_df = engineer_features(train_df, config)
    test_df = engineer_features(test_df, config)

    # Step 4: Model Training
    logger.info("-" * 40)
    logger.info("STEP 4: Model Training")
    logger.info("-" * 40)
    models, val_df, feature_names = train_models(train_df, config)

    # Step 5: Evaluation
    logger.info("-" * 40)
    logger.info("STEP 5: Evaluation")
    logger.info("-" * 40)
    metrics = evaluate_models(models, val_df, feature_names, config)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info("=" * 60)

    for model_name, model_metrics in metrics.items():
        logger.info(
            f"  {model_name}: RMSE={model_metrics['rmse']:.2f}, "
            f"MAE={model_metrics['mae']:.2f}, "
            f"R²={model_metrics['r2_score']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Predictive Maintenance Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    run(args.config)
