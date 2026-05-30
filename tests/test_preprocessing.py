"""
Tests for the preprocessing module.

Validates that RUL capping, sensor dropping, and normalization
work correctly on synthetic data.
"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessing import add_rul_column, drop_uninformative_sensors, normalize_sensors


@pytest.fixture
def sample_train_data():
    """Create a small synthetic training dataset mimicking C-MAPSS format."""
    data = {
        "engine_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
        "op_setting_1": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1],
        "op_setting_2": [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
        "op_setting_3": [100, 100, 100, 100, 100, 100, 100, 100],  # constant
        "sensor_1": [500, 500, 500, 500, 500, 500, 500, 500],  # constant
        "sensor_2": [641, 642, 643, 644, 645, 640, 641, 642],
        "sensor_3": [1580, 1582, 1585, 1590, 1600, 1579, 1581, 1583],
    }
    return pd.DataFrame(data)


class TestAddRulColumn:
    """Tests for the RUL target creation."""

    def test_rul_values_are_correct(self, sample_train_data):
        """RUL should count down from max_cycle to 0 for each engine."""
        df = add_rul_column(sample_train_data, max_rul=125)

        # Engine 1 has 5 cycles: RUL = 4, 3, 2, 1, 0
        engine_1 = df[df["engine_id"] == 1]["rul"].tolist()
        assert engine_1 == [4, 3, 2, 1, 0]

        # Engine 2 has 3 cycles: RUL = 2, 1, 0
        engine_2 = df[df["engine_id"] == 2]["rul"].tolist()
        assert engine_2 == [2, 1, 0]

    def test_rul_is_capped(self, sample_train_data):
        """RUL should never exceed max_rul."""
        df = add_rul_column(sample_train_data, max_rul=3)

        engine_1 = df[df["engine_id"] == 1]["rul"].tolist()
        assert engine_1 == [3, 3, 2, 1, 0]  # First value capped at 3

    def test_rul_column_exists(self, sample_train_data):
        """Output should contain a 'rul' column."""
        df = add_rul_column(sample_train_data, max_rul=125)
        assert "rul" in df.columns

    def test_last_cycle_rul_is_zero(self, sample_train_data):
        """The last cycle of each engine should have RUL = 0."""
        df = add_rul_column(sample_train_data, max_rul=125)

        for engine_id in df["engine_id"].unique():
            engine = df[df["engine_id"] == engine_id]
            last_rul = engine.iloc[-1]["rul"]
            assert last_rul == 0, f"Engine {engine_id} last RUL should be 0"


class TestDropUninformativeSensors:
    """Tests for dropping constant/useless sensors."""

    def test_drops_specified_sensors(self, sample_train_data):
        """Specified sensors should be removed."""
        df = drop_uninformative_sensors(sample_train_data, drop_sensors=[1], drop_settings=[3])

        assert "sensor_1" not in df.columns
        assert "op_setting_3" not in df.columns

    def test_keeps_informative_sensors(self, sample_train_data):
        """Non-specified sensors should be kept."""
        df = drop_uninformative_sensors(sample_train_data, drop_sensors=[1], drop_settings=[3])

        assert "sensor_2" in df.columns
        assert "sensor_3" in df.columns
        assert "op_setting_1" in df.columns

    def test_handles_nonexistent_sensor(self, sample_train_data):
        """Dropping a sensor that doesn't exist should not raise an error."""
        df = drop_uninformative_sensors(sample_train_data, drop_sensors=[99], drop_settings=[])
        assert len(df.columns) == len(sample_train_data.columns)


class TestNormalizeSensors:
    """Tests for sensor normalization."""

    def test_minmax_range(self, sample_train_data):
        """MinMax scaled values should be between 0 and 1."""
        df, scaler = normalize_sensors(sample_train_data, scaler_type="minmax", fit=True)

        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        for col in sensor_cols:
            if df[col].std() > 0:  # Skip constant columns
                assert df[col].min() >= -0.001, f"{col} min below 0"
                assert df[col].max() <= 1.001, f"{col} max above 1"

    def test_scaler_is_returned(self, sample_train_data):
        """A fitted scaler object should be returned."""
        _, scaler = normalize_sensors(sample_train_data, scaler_type="minmax", fit=True)
        assert scaler is not None
        assert hasattr(scaler, "transform")
