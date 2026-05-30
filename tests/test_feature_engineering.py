"""
Tests for the feature engineering module.

Validates that rolling features and slope features are computed
correctly on synthetic data.
"""

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import add_rolling_features, add_slope_features, get_sensor_columns


@pytest.fixture
def sample_sensor_data():
    """Create synthetic sensor data for two engines."""
    data = {
        "engine_id": [1]*10 + [2]*8,
        "cycle": list(range(1, 11)) + list(range(1, 9)),
        "sensor_2": list(range(100, 110)) + list(range(200, 208)),
        "sensor_3": [50 + i*2 for i in range(10)] + [80 + i*3 for i in range(8)],
    }
    return pd.DataFrame(data)


class TestGetSensorColumns:
    """Tests for sensor column detection."""

    def test_finds_sensor_columns(self, sample_sensor_data):
        """Should identify all columns starting with 'sensor_'."""
        cols = get_sensor_columns(sample_sensor_data)
        assert "sensor_2" in cols
        assert "sensor_3" in cols
        assert "engine_id" not in cols
        assert "cycle" not in cols


class TestRollingFeatures:
    """Tests for rolling window feature generation."""

    def test_creates_correct_column_names(self, sample_sensor_data):
        """Rolling feature columns should follow naming convention."""
        df = add_rolling_features(
            sample_sensor_data,
            window_sizes=[3],
            rolling_features=["mean"],
        )
        assert "sensor_2_rolling_mean_3" in df.columns
        assert "sensor_3_rolling_mean_3" in df.columns

    def test_number_of_new_features(self, sample_sensor_data):
        """Should create n_sensors * n_windows * n_features new columns."""
        df = add_rolling_features(
            sample_sensor_data,
            window_sizes=[3, 5],
            rolling_features=["mean", "std"],
        )
        # 2 sensors * 2 windows * 2 features = 8 new columns
        new_cols = len(df.columns) - len(sample_sensor_data.columns)
        assert new_cols == 8

    def test_rolling_mean_values(self, sample_sensor_data):
        """Rolling mean should be correct for a known sequence."""
        df = add_rolling_features(
            sample_sensor_data,
            window_sizes=[3],
            rolling_features=["mean"],
        )
        # Engine 1, sensor_2 = [100, 101, 102, ...], window=3
        # At cycle 3: mean of [100, 101, 102] = 101.0
        engine_1 = df[df["engine_id"] == 1].reset_index(drop=True)
        rolling_mean_at_cycle_3 = engine_1.loc[2, "sensor_2_rolling_mean_3"]
        assert abs(rolling_mean_at_cycle_3 - 101.0) < 0.01


class TestSlopeFeatures:
    """Tests for rate-of-change feature generation."""

    def test_creates_slope_columns(self, sample_sensor_data):
        """Should add slope columns for each sensor."""
        df = add_slope_features(sample_sensor_data, slope_window=3)
        assert "sensor_2_slope_3" in df.columns
        assert "sensor_3_slope_3" in df.columns

    def test_no_nans_after_slope(self, sample_sensor_data):
        """NaN values from diff should be filled with 0."""
        df = add_slope_features(sample_sensor_data, slope_window=3)
        slope_cols = [c for c in df.columns if "_slope_" in c]
        for col in slope_cols:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_slope_value_for_linear_sensor(self, sample_sensor_data):
        """For a linearly increasing sensor, slope should be constant."""
        df = add_slope_features(sample_sensor_data, slope_window=3)
        engine_1 = df[df["engine_id"] == 1].reset_index(drop=True)
        # sensor_2 increases by 1 each cycle, slope_window=3
        # slope = (current - 3_cycles_ago) / 3 = 3/3 = 1.0
        # This applies from cycle 4 onward (index 3+)
        slope_val = engine_1.loc[5, "sensor_2_slope_3"]
        assert abs(slope_val - 1.0) < 0.01
