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
        
    def get_activation_energy(self, X: np.ndarray) -> np.ndarray:
        """Retrieve the predicted activation energy (Ea) if applicable.
        
        Design Intent: Physics-Informed Foundation Layer.
        Models should expose thermodynamic parameters like Ea instead of just
        black-box rates (k). This enables Arrhenius-consistent simulation
        and thermodynamic validation.
        
        Returns:
            Activation energy values.
            Raises NotImplementedError if the model is purely data-driven
            and does not extract physical parameters.
        """
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not support "
            f"activation energy extraction."
        )
        
    def validate_thermodynamics(self, state: Dict[str, Any]) -> bool:
        """Validate if the current predicted rates obey thermodynamic constraints.
        
        Design Intent: Provides a unified hook for checking physical consistency
        (e.g., microscopic reversibility, non-negative rate constants) across
        all kinetic models before simulation stepping.
        
        Args:
            state: Current thermodynamic state (T, P, concentrations)
            
        Returns:
            True if thermodynamically valid, False otherwise.
        """
        # Default implementation assumes validity, overridden by specific models
        return True

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
