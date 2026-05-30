"""
Training Module
===============
Trains regression models to predict Remaining Useful Life (RUL).
Supports Random Forest and XGBoost, with all hyperparameters
driven from config.yaml. Models are saved as joblib artifacts.
"""

from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from src.utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Drops non-feature columns: engine_id, cycle, rul.

    Args:
        df: Featured DataFrame.

    Returns:
        Tuple of (feature matrix, target series).
    """
    drop_cols = ["engine_id", "cycle", "rul"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["rul"]

    return X, y


def split_by_engine(
    df: pd.DataFrame, val_split: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation sets by engine (no data leakage).

    Splitting by engine ensures that cycles from the same engine don't
    appear in both train and validation sets — which would cause leakage
    since consecutive cycles are highly correlated.

    Args:
        df: Full training DataFrame.
        val_split: Fraction of engines for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_split, random_state=random_state
    )
    groups = df["engine_id"]

    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    n_train_engines = train_df["engine_id"].nunique()
    n_val_engines = val_df["engine_id"].nunique()

    logger.info(
        f"Split: {n_train_engines} train engines ({len(train_df)} rows), "
        f"{n_val_engines} val engines ({len(val_df)} rows)"
    )

    return train_df, val_df


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> RandomForestRegressor:
    """Train a Random Forest regressor."""
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete")
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    """Train an XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("XGBoost not installed, skipping")
        return None

    logger.info("Training XGBoost...")
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost training complete")
    return model


def train_models(
    train_df: pd.DataFrame, config: dict
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Full training pipeline: split data, train models, save artifacts.

    Args:
        train_df: Featured training DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (models dict, val_df for evaluation, feature names).
    """
    train_config = config["training"]
    results_dir = ensure_dir(config["evaluation"]["output_dir"])

    # Split by engine
    train_split, val_split = split_by_engine(
        train_df, train_config["val_split"], train_config["random_state"]
    )

    X_train, y_train = get_feature_target_split(train_split)
    X_val, y_val = get_feature_target_split(val_split)

    models = {}

    # Train Random Forest
    rf_model = train_random_forest(
        X_train, y_train, train_config["models"]["random_forest"]
    )
    models["random_forest"] = rf_model

    # Train XGBoost
    xgb_model = train_xgboost(
        X_train, y_train, train_config["models"]["xgboost"]
    )
    if xgb_model is not None:
        models["xgboost"] = xgb_model

    # Save models
    for name, model in models.items():
        model_path = results_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to {model_path}")

    # Save feature names for reproducibility
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, results_dir / "feature_names.joblib")

    return models, val_split, feature_names


if __name__ == "__main__":
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train_featured.csv")
    models, val_df, feature_names = train_models(train_df, config)

    logger.info(f"Trained {len(models)} models: {list(models.keys())}")
