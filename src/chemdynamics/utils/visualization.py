"""Visualization utilities for model results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


class ReactionVisualizer:
    """Visualization tools for reaction prediction results."""

    @staticmethod
    def plot_cross_validation_results(
        cv_results: Dict[str, np.ndarray],
        save_path: Optional[Path] = None,
        title: str = "Cross-Validation Results",
    ):
        """Plot cross-validation scores for multiple models.

        Args:
            cv_results: Dictionary mapping model names to CV scores
            save_path: Optional path to save the figure
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(cv_results)
        sns.boxplot(data=df)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.ylabel("R² Score", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None,
    ):
        """Plot predicted vs actual values.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", s=50)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        plt.title(f"{model_name}: Predicted vs Actual", fontsize=14, fontweight="bold")
        plt.xlabel("Actual Reaction Rate (mol/L·s)", fontsize=12)
        plt.ylabel("Predicted Reaction Rate (mol/L·s)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_feature_importance(
        importance: np.ndarray,
        feature_names: list,
        model_name: str = "Model",
        save_path: Optional[Path] = None,
    ):
        """Plot feature importance.

        Args:
            importance: Feature importance values
            feature_names: Names of features
            model_name: Name of the model
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title(
            f"{model_name}: Feature Importance", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None,
    ):
        """Plot residuals to check for patterns.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Optional path to save the figure
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 5))

        # Residuals vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
        plt.axhline(y=0, color="r", linestyle="--", lw=2)
        plt.title("Residuals vs Predicted Values")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)

        # Residual histogram
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        plt.suptitle(f"{model_name}: Residual Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        plt.show()
