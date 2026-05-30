"""
Preprocessing Module
====================
Cleans raw sensor data, drops uninformative sensors, normalizes readings,
and creates the piecewise-linear RUL target variable.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)


def add_rul_column(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) target to training data.

    Uses a piecewise linear degradation model: RUL is capped at max_rul
    because engines in early life show no measurable degradation, so
    predicting exact large RUL values adds noise without value.

    Args:
        df: Training DataFrame with engine_id and cycle columns.
        max_rul: Maximum RUL cap (default 125).

    Returns:
        DataFrame with 'rul' column added.
    """
    # For each engine, max cycle = failure point → RUL at that point is 0
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    # Apply piecewise cap
    df["rul"] = df["rul"].clip(upper=max_rul)
    logger.info(f"RUL column added with cap at {max_rul} cycles")

    return df


def drop_uninformative_sensors(
    df: pd.DataFrame, drop_sensors: list[int], drop_settings: list[int]
) -> pd.DataFrame:
    """
    Remove sensors and settings with near-zero variance.

    In FD001, sensors 1, 5, 6, 10, 16, 18, 19 and operational setting 3
    have constant or near-constant values across all engines — they carry
    no degradation signal.

    Args:
        df: Input DataFrame.
        drop_sensors: List of sensor numbers to drop.
        drop_settings: List of operational setting numbers to drop.

    Returns:
        DataFrame with uninformative columns removed.
    """
    cols_to_drop = [f"sensor_{s}" for s in drop_sensors]
    cols_to_drop += [f"op_setting_{s}" for s in drop_settings]

    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped {len(existing)} uninformative columns: {existing}")

    return df


def normalize_sensors(
    df: pd.DataFrame, scaler_type: str = "minmax", fit: bool = True, scaler=None
) -> tuple[pd.DataFrame, object]:
    """
    Normalize sensor readings across the dataset.

    Args:
        df: Input DataFrame.
        scaler_type: "minmax" or "standard".
        fit: Whether to fit the scaler (True for train, False for test).
        scaler: Pre-fitted scaler (used when fit=False).

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    setting_cols = [c for c in df.columns if c.startswith("op_setting_")]
    cols_to_scale = sensor_cols + setting_cols

    if scaler_type == "minmax":
        scaler = scaler or MinMaxScaler()
    else:
        scaler = scaler or StandardScaler()

    if fit:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"Fitted {scaler_type} scaler on {len(cols_to_scale)} columns")
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        logger.info(f"Applied pre-fitted scaler on {len(cols_to_scale)} columns")

    return df, scaler


def preprocess_train(
    train_df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, object]:
    """
    Full preprocessing pipeline for training data.

    Args:
        train_df: Raw training DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler).
    """
    prep_config = config["preprocessing"]

    logger.info("Starting training data preprocessing")

    # Add RUL target
    train_df = add_rul_column(train_df, max_rul=prep_config["max_rul"])

    # Drop uninformative columns
    train_df = drop_uninformative_sensors(
        train_df, prep_config["drop_sensors"], prep_config["drop_settings"]
    )

    # Normalize
    train_df, scaler = normalize_sensors(
        train_df, scaler_type=prep_config["scaler_type"], fit=True
    )

    logger.info(
        f"Preprocessing complete: {train_df.shape[0]} rows, "
        f"{train_df.shape[1]} columns"
    )

    return train_df, scaler


def preprocess_test(
    test_df: pd.DataFrame, rul_df: pd.DataFrame, config: dict, scaler
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for test data.

    Args:
        test_df: Raw test DataFrame.
        rul_df: Ground truth RUL labels.
        config: Configuration dictionary.
        scaler: Pre-fitted scaler from training.

    Returns:
        Preprocessed test DataFrame.
    """
    prep_config = config["preprocessing"]

    logger.info("Starting test data preprocessing")

    # Drop uninformative columns
    test_df = drop_uninformative_sensors(
        test_df, prep_config["drop_sensors"], prep_config["drop_settings"]
    )

    # Normalize using training scaler
    test_df, _ = normalize_sensors(
        test_df, scaler_type=prep_config["scaler_type"], fit=False, scaler=scaler
    )

    # Add RUL for the last cycle of each engine (from ground truth)
    max_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    test_df = test_df.merge(max_cycles, on="engine_id", how="left")

    rul_map = rul_df.set_index("engine_id")["rul"].to_dict()
    test_df["rul"] = test_df.apply(
        lambda row: rul_map.get(row["engine_id"], 0)
        + (row["max_cycle"] - row["cycle"]),
        axis=1,
    )
    test_df["rul"] = test_df["rul"].clip(upper=prep_config["max_rul"])
    test_df.drop("max_cycle", axis=1, inplace=True)

    logger.info(
        f"Test preprocessing complete: {test_df.shape[0]} rows, "
        f"{test_df.shape[1]} columns"
    )

    return test_df


if __name__ == "__main__":
    config = load_config()
    processed_dir = ensure_dir(config["data"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train_raw.csv")
    test_df = pd.read_csv(processed_dir / "test_raw.csv")
    rul_df = pd.read_csv(processed_dir / "rul_labels.csv")

    train_df, scaler = preprocess_train(train_df, config)
    test_df = preprocess_test(test_df, rul_df, config, scaler)

    train_df.to_csv(processed_dir / "train_preprocessed.csv", index=False)
    test_df.to_csv(processed_dir / "test_preprocessed.csv", index=False)
