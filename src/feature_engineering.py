"""
Feature Engineering Module
==========================
Creates time-series features from sensor data: rolling window statistics,
rate-of-change (slope) features, and cumulative operational metrics.

Raw sensor readings at a single time point tell you the current state.
These engineered features capture degradation *trends* — which are far
more predictive of remaining useful life.
"""

import pandas as pd

from src.utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return list of sensor column names present in the DataFrame."""
    return [c for c in df.columns if c.startswith("sensor_")]


def add_rolling_features(
    df: pd.DataFrame,
    window_sizes: list[int],
    rolling_features: list[str],
) -> pd.DataFrame:
    """
    Add rolling window statistics per engine for each sensor.

    For each sensor and each window size, computes statistical aggregates
    (mean, std, min, max) over the trailing window of cycles. This captures
    how sensor readings are changing over time — a sensor whose rolling
    standard deviation is increasing indicates growing instability.

    Args:
        df: Preprocessed DataFrame sorted by engine_id and cycle.
        window_sizes: List of window sizes (e.g., [5, 10, 20]).
        rolling_features: List of stats to compute (e.g., ["mean", "std"]).

    Returns:
        DataFrame with rolling features added.
    """
    sensor_cols = get_sensor_columns(df)
    new_features = []

    for window in window_sizes:
        logger.info(f"Computing rolling features for window={window}")

        grouped = df.groupby("engine_id")[sensor_cols]

        for feat in rolling_features:
            rolled = getattr(grouped.rolling(window=window, min_periods=1), feat)()
            rolled = rolled.reset_index(level=0, drop=True)
            rolled.columns = [
                f"{col}_rolling_{feat}_{window}" for col in sensor_cols
            ]
            new_features.append(rolled)

    result = pd.concat([df] + new_features, axis=1)
    logger.info(
        f"Added {len(result.columns) - len(df.columns)} rolling features"
    )

    return result


def add_slope_features(df: pd.DataFrame, slope_window: int = 10) -> pd.DataFrame:
    """
    Add rate-of-change features for each sensor.

    The slope approximates the first derivative of sensor readings over
    a trailing window. A negative slope on a health indicator means the
    engine is actively degrading — this is highly predictive of near-term
    failure.

    Slope is calculated as the difference between the current value and
    the value `slope_window` cycles ago, divided by the window size.

    Args:
        df: DataFrame with sensor columns.
        slope_window: Number of cycles to look back for slope calculation.

    Returns:
        DataFrame with slope features added.
    """
    sensor_cols = get_sensor_columns(df)

    for col in sensor_cols:
        slope_col = f"{col}_slope_{slope_window}"
        df[slope_col] = df.groupby("engine_id")[col].diff(slope_window) / slope_window

    # Fill NaN slopes (first few cycles per engine) with 0
    slope_cols = [c for c in df.columns if "_slope_" in c]
    df[slope_cols] = df[slope_cols].fillna(0)

    logger.info(f"Added {len(slope_cols)} slope features (window={slope_window})")

    return df


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with all engineered features.
    """
    fe_config = config["feature_engineering"]

    logger.info("Starting feature engineering")

    # Ensure sorted by engine and cycle
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # Rolling window features
    df = add_rolling_features(
        df,
        window_sizes=fe_config["window_sizes"],
        rolling_features=fe_config["rolling_features"],
    )

    # Slope features
    if fe_config.get("add_slope", True):
        df = add_slope_features(df, slope_window=fe_config["slope_window"])

    # Drop any remaining NaN rows (from rolling calculations)
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_rows - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN values after feature engineering")

    logger.info(
        f"Feature engineering complete: {df.shape[1]} total columns "
        f"({df.shape[0]} rows)"
    )

    return df


if __name__ == "__main__":
    config = load_config()
    processed_dir = ensure_dir(config["data"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train_preprocessed.csv")
    test_df = pd.read_csv(processed_dir / "test_preprocessed.csv")

    train_fe = engineer_features(train_df, config)
    test_fe = engineer_features(test_df, config)

    train_fe.to_csv(processed_dir / "train_featured.csv", index=False)
    test_fe.to_csv(processed_dir / "test_featured.csv", index=False)
    logger.info("Featured datasets saved")
