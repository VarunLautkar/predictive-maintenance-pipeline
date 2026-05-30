"""
Data Ingestion Module
=====================
Loads raw NASA C-MAPSS text files and converts them into structured DataFrames.
The raw files have no headers — this module assigns proper column names,
validates the data, and saves clean CSVs for downstream processing.
"""

from pathlib import Path

import pandas as pd

from src.utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)

# Column names for the C-MAPSS dataset
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_raw_file(filepath: str, columns: list[str]) -> pd.DataFrame:
    """
    Load a single raw C-MAPSS text file into a DataFrame.

    The raw files are space-delimited with no headers and trailing spaces
    that create extra NaN columns — these are stripped automatically.

    Args:
        filepath: Path to the raw text file.
        columns: List of column names to assign.

    Returns:
        Clean DataFrame with proper column names.
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")

    # Raw files sometimes have trailing whitespace creating extra columns
    df = df.iloc[:, : len(columns)]
    df.columns = columns

    return df


def load_rul_file(filepath: str) -> pd.DataFrame:
    """
    Load the RUL (Remaining Useful Life) ground truth file.

    Args:
        filepath: Path to the RUL text file.

    Returns:
        DataFrame with engine_id and true RUL values.
    """
    rul = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    rul.columns = ["rul"]
    rul["engine_id"] = rul.index + 1
    return rul[["engine_id", "rul"]]


def ingest_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full data ingestion pipeline: load all raw files and save as CSVs.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (train_df, test_df, rul_df).
    """
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = ensure_dir(config["data"]["processed_dir"])

    # Load training data
    train_path = raw_dir / config["data"]["train_file"]
    logger.info(f"Loading training data from {train_path}")
    train_df = load_raw_file(str(train_path), COLUMN_NAMES)
    logger.info(
        f"Training data: {train_df['engine_id'].nunique()} engines, "
        f"{len(train_df)} total rows"
    )

    # Load test data
    test_path = raw_dir / config["data"]["test_file"]
    logger.info(f"Loading test data from {test_path}")
    test_df = load_raw_file(str(test_path), COLUMN_NAMES)
    logger.info(
        f"Test data: {test_df['engine_id'].nunique()} engines, "
        f"{len(test_df)} total rows"
    )

    # Load RUL ground truth
    rul_path = raw_dir / config["data"]["rul_file"]
    logger.info(f"Loading RUL labels from {rul_path}")
    rul_df = load_rul_file(str(rul_path))
    logger.info(f"RUL labels loaded for {len(rul_df)} engines")

    # Save processed CSVs
    train_df.to_csv(processed_dir / "train_raw.csv", index=False)
    test_df.to_csv(processed_dir / "test_raw.csv", index=False)
    rul_df.to_csv(processed_dir / "rul_labels.csv", index=False)
    logger.info(f"Raw data saved to {processed_dir}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    config = load_config()
    ingest_data(config)
