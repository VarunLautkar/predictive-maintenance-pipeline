"""
Evaluation Module
=================
Evaluates trained models on validation/test data. Computes standard regression
metrics (RMSE, MAE, R²) plus an asymmetric scoring function that penalizes
late predictions more heavily — because a late RUL prediction (engine fails
before you predicted) is far more dangerous than an early one.

Generates evaluation plots and saves all results to JSON.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.train import get_feature_target_split
from src.utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics.

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dictionary of metric names and values.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }


def asymmetric_score(
    y_true: np.ndarray, y_pred: np.ndarray, late_penalty: float = 2.0
) -> float:
    """
    Compute asymmetric error score.

    Late predictions (predicted RUL > actual RUL) are penalized more because
    they mean the engine failed before we expected. Early predictions waste
    maintenance resources but don't cause failures.

    Args:
        y_true: Actual RUL values.
        y_pred: Predicted RUL values.
        late_penalty: Multiplier for late prediction errors.

    Returns:
        Weighted mean absolute error.
    """
    errors = y_pred - y_true
    weights = np.where(errors > 0, late_penalty, 1.0)
    return float(np.mean(weights * np.abs(errors)))


def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: Path
) -> None:
    """Generate scatter plot of predicted vs actual RUL."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2196F3")
    ax.plot(
        [0, max(y_true)], [0, max(y_true)],
        "r--", linewidth=2, label="Perfect Prediction"
    )

    ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title(f"{model_name} — Predicted vs Actual RUL", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved prediction scatter plot for {model_name}")


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: Path
) -> None:
    """Generate residual distribution plot."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(residuals, bins=50, color="#4CAF50", alpha=0.7, edgecolor="black")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (Predicted - Actual)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title(f"{model_name} — Residual Distribution", fontsize=13)

    # Residuals vs Actual
    axes[1].scatter(y_true, residuals, alpha=0.3, s=10, color="#FF9800")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Actual RUL (cycles)", fontsize=11)
    axes[1].set_ylabel("Residual", fontsize=11)
    axes[1].set_title(f"{model_name} — Residuals vs Actual RUL", fontsize=13)

    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_residuals.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved residual plots for {model_name}")


def plot_feature_importance(
    model, feature_names: list[str], model_name: str, output_dir: Path, top_n: int = 20
) -> None:
    """Generate feature importance bar chart (top N features)."""
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} does not support feature importances")
        return

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
    ax.barh(
        feat_imp["feature"][::-1],
        feat_imp["importance"][::-1],
        color=colors,
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / f"{model_name}_feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved feature importance plot for {model_name}")


def evaluate_models(
    models: dict,
    val_df: pd.DataFrame,
    feature_names: list[str],
    config: dict,
) -> dict:
    """
    Full evaluation pipeline for all trained models.

    Args:
        models: Dictionary of {model_name: trained_model}.
        val_df: Validation DataFrame.
        feature_names: List of feature column names.
        config: Configuration dictionary.

    Returns:
        Dictionary of all model metrics.
    """
    eval_config = config["evaluation"]
    output_dir = ensure_dir(eval_config["output_dir"])
    late_penalty = eval_config.get("late_penalty_factor", 2.0)

    X_val, y_val = get_feature_target_split(val_df)
    y_true = y_val.values

    all_metrics = {}

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        y_pred = model.predict(X_val)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        metrics["asymmetric_score"] = asymmetric_score(y_true, y_pred, late_penalty)
        metrics["n_late_predictions"] = int(np.sum(y_pred > y_true))
        metrics["n_early_predictions"] = int(np.sum(y_pred < y_true))

        all_metrics[name] = metrics

        logger.info(
            f"  {name}: RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, R²={metrics['r2_score']:.3f}"
        )

        # Generate plots
        plot_predictions_vs_actual(y_true, y_pred, name, output_dir)
        plot_residuals(y_true, y_pred, name, output_dir)
        plot_feature_importance(model, feature_names, name, output_dir)

    # Save metrics to JSON
    metrics_path = output_dir / eval_config["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"All metrics saved to {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    import joblib

    config = load_config()
    results_dir = Path(config["evaluation"]["output_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    # Load models and data
    feature_names = joblib.load(results_dir / "feature_names.joblib")
    val_df = pd.read_csv(processed_dir / "train_featured.csv")

    models = {}
    for model_file in results_dir.glob("*_model.joblib"):
        name = model_file.stem.replace("_model", "")
        models[name] = joblib.load(model_file)

    evaluate_models(models, val_df, feature_names, config)
