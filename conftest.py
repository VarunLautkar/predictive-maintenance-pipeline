"""
conftest.py — Root-level pytest configuration.

Ensures that the project root directory is on sys.path so that
`from src.xxx import yyy` works without needing to set PYTHONPATH manually.
"""
import sys
from pathlib import Path

# Add the project root to sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).parent))
