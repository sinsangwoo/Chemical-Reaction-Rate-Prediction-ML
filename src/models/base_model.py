"""Base class for all prediction models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class BaseReactionModel(ABC):
    """Abstract base class for reaction rate prediction models."""

    def __init__(self, model_config: Dict[str, Any] = None):
        """Initialize the model.

        Args:
            model_config: Configuration dictionary for the model
        """
        self.model_config = model_config or {}
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predicted reaction rates
        """
        pass

    def save_model(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        import joblib

        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load the model from
        """
        import joblib

        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
