"""Tests for evaluation metrics."""

import pytest
import numpy as np
from src.evaluation.metrics import RegressionMetrics


class TestRegressionMetrics:
    """Test suite for RegressionMetrics."""

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)

        assert metrics["MAE"] == 0
        assert metrics["MSE"] == 0
        assert metrics["RMSE"] == 0
        assert metrics["R2"] == 1.0
        assert metrics["MAPE"] == 0

    def test_known_values(self):
        """Test metrics with known error values."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 6])  # Last prediction off by 1
        metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)

        assert metrics["MAE"] == 0.2  # Average error of 1/5
        assert metrics["MSE"] == 0.2  # (1^2)/5
        assert np.isclose(metrics["RMSE"], np.sqrt(0.2))

    def test_metrics_keys(self):
        """Test that all expected metrics are present."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 3.1])
        metrics = RegressionMetrics.calculate_metrics(y_true, y_pred)

        expected_keys = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
        assert all(key in metrics for key in expected_keys)
