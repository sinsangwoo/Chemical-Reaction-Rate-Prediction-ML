"""Evaluation metrics for regression models."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from typing import Dict


class RegressionMetrics:
    """Calculate and display regression metrics."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
        """Print metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{'='*50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*50}")
        for metric_name, value in metrics.items():
            print(f"{metric_name:10s}: {value:.6f}")
        print(f"{'='*50}\n")

    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> None:
        """Compare multiple models side by side.

        Args:
            results: Dictionary mapping model names to their metrics
        """
        import pandas as pd

        df = pd.DataFrame(results).T
        print("\nModel Comparison:")
        print(df.to_string())
        print(f"\nBest model by R2: {df['R2'].idxmax()} (R2={df['R2'].max():.4f})")
        print(f"Best model by MAE: {df['MAE'].idxmin()} (MAE={df['MAE'].min():.4f})")
