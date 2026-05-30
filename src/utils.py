"""Utility functions: logging setup, config loading, and helpers."""

import logging
import sys
from pathlib import Path

import yaml


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with console output.

    Args:
        name: Logger name (typically module name).
        level: Logging level (default INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(directory: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
