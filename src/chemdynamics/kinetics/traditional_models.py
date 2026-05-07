"""Traditional ML models: Linear, Polynomial, RandomForest, XGBoost."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
from .base_model import BaseReactionModel


class LinearRegressionModel(BaseReactionModel):
    """Linear regression model for reaction rate prediction."""

    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(model_config)
        self.model = LinearRegression(**self.model_config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train linear regression model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using linear regression."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)


class PolynomialRegressionModel(BaseReactionModel):
    """Polynomial regression model."""

    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(model_config)
        degree = self.model_config.get("degree", 2)
        self.model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear_regression", LinearRegression()),
            ]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train polynomial regression model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using polynomial regression."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)


class SVRModel(BaseReactionModel):
    """Support Vector Regression model."""

    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(model_config)
        default_config = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
        config = {**default_config, **self.model_config}
        self.model = SVR(**config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train SVR model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using SVR."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)


class RandomForestModel(BaseReactionModel):
    """Random Forest regression model."""

    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(model_config)
        default_config = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
        config = {**default_config, **self.model_config}
        self.model = RandomForestRegressor(**config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train Random Forest model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Random Forest."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.feature_importances_
