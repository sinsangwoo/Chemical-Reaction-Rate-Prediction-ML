"""Tests for model implementations."""

import pytest
import numpy as np
from src.models.traditional_models import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    RandomForestModel,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.rand(100, 3)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
    X_test = np.random.rand(20, 3)
    return X_train, y_train, X_test


class TestLinearRegressionModel:
    """Test suite for LinearRegressionModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = LinearRegressionModel()
        assert model.model is not None
        assert not model.is_trained

    def test_training(self, sample_data):
        """Test model training."""
        X_train, y_train, _ = sample_data
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        assert model.is_trained

    def test_prediction(self, sample_data):
        """Test model prediction."""
        X_train, y_train, X_test = sample_data
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_prediction_without_training(self, sample_data):
        """Test prediction raises error without training."""
        _, _, X_test = sample_data
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X_test)


class TestRandomForestModel:
    """Test suite for RandomForestModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestModel()
        assert model.model is not None
        assert not model.is_trained

    def test_custom_config(self):
        """Test model with custom configuration."""
        config = {"n_estimators": 50, "max_depth": 10}
        model = RandomForestModel(model_config=config)
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 10

    def test_training(self, sample_data):
        """Test model training."""
        X_train, y_train, _ = sample_data
        model = RandomForestModel()
        model.train(X_train, y_train)
        assert model.is_trained

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, y_train, _ = sample_data
        model = RandomForestModel()
        model.train(X_train, y_train)
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        assert np.isclose(importance.sum(), 1.0)
