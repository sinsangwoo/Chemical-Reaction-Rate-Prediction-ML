"""Conformal Prediction for distribution-free uncertainty quantification.

Provides rigorous statistical guarantees on prediction sets without
assuming any model or data distribution.

Key Advantages:
- Distribution-free: No assumptions about data or model
- Finite-sample guarantees: Valid for any sample size
- Model-agnostic: Works with any prediction model
- Adaptive: Adjusts to model confidence

Based on:
- Cha et al. (2023): Conformal Prediction for Bayesian GNN
- Angelopoulos & Bates (2021): Gentle Introduction to Conformal Prediction
- NeurIPS 2023: Temperature scaling for GNN
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConformalPrediction:
    """Conformal prediction result.
    
    Attributes:
        point_estimate: Point prediction
        prediction_interval: (lower, upper) interval
        coverage: Target coverage probability (e.g., 0.95)
        interval_width: Width of prediction interval
    """
    point_estimate: float
    prediction_interval: Tuple[float, float]
    coverage: float
    interval_width: float
    
    def __repr__(self):
        lower, upper = self.prediction_interval
        return (
            f"ConformalPrediction(\n"
            f"  point={self.point_estimate:.4f},\n"
            f"  interval=[{lower:.4f}, {upper:.4f}],\n"
            f"  coverage={self.coverage:.1%},\n"
            f"  width={self.interval_width:.4f}\n"
            f")"
        )


class ConformalPredictor:
    """Conformal prediction wrapper for any model.
    
    Implements split conformal prediction:
    1. Fit model on training set
    2. Compute nonconformity scores on calibration set
    3. Use quantile of scores to construct prediction intervals
    
    Args:
        model: Trained prediction model
        alpha: Miscoverage rate (1 - coverage), e.g., 0.05 for 95%
    """
    
    def __init__(self, model, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha
        self.coverage = 1 - alpha
        self.quantile = None
    
    def calibrate(self, cal_data: List, cal_targets: List):
        """Calibrate on calibration set.
        
        Args:
            cal_data: List of calibration samples
            cal_targets: True values for calibration set
        """
        # Compute nonconformity scores
        scores = []
        
        for data, target in zip(cal_data, cal_targets):
            # Get prediction
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(data).item()
            else:
                pred = self.model.predict(data)
            
            # Nonconformity score: absolute residual
            score = abs(pred - target)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Compute quantile (with finite-sample correction)
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, q)
    
    def predict(self, data) -> ConformalPrediction:
        """Make conformal prediction.
        
        Args:
            data: Input sample
        
        Returns:
            ConformalPrediction with interval guarantees
        """
        if self.quantile is None:
            raise ValueError("Must calibrate before predicting")
        
        # Get point prediction
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                point = self.model(data).item()
        else:
            point = self.model.predict(data)
        
        # Construct prediction interval
        lower = point - self.quantile
        upper = point + self.quantile
        width = 2 * self.quantile
        
        return ConformalPrediction(
            point_estimate=point,
            prediction_interval=(lower, upper),
            coverage=self.coverage,
            interval_width=width
        )


class AdaptiveConformalPredictor(ConformalPredictor):
    """Adaptive conformal prediction with local calibration.
    
    Adjusts prediction intervals based on local difficulty:
    - Narrow intervals for easy samples (low uncertainty)
    - Wide intervals for hard samples (high uncertainty)
    
    Based on:
    - Gendler et al. (2023): Adaptive Conformal Prediction
    - Lei et al. (2018): Distribution-free prediction sets
    """
    
    def __init__(self, model, alpha: float = 0.05, difficulty_estimator=None):
        super().__init__(model, alpha)
        self.difficulty_estimator = difficulty_estimator
        self.score_function = None
    
    def calibrate(self, cal_data: List, cal_targets: List, cal_features: Optional[List] = None):
        """Calibrate with local difficulty.
        
        Args:
            cal_data: Calibration samples
            cal_targets: True values
            cal_features: Additional features for difficulty estimation
        """
        scores = []
        difficulties = []
        
        for i, (data, target) in enumerate(zip(cal_data, cal_targets)):
            # Get prediction
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(data).item()
            else:
                pred = self.model.predict(data)
            
            # Nonconformity score
            score = abs(pred - target)
            scores.append(score)
            
            # Estimate difficulty (e.g., distance to training data)
            if self.difficulty_estimator:
                features = cal_features[i] if cal_features else None
                difficulty = self.difficulty_estimator(data, features)
            else:
                difficulty = 1.0  # Uniform difficulty
            
            difficulties.append(difficulty)
        
        scores = np.array(scores)
        difficulties = np.array(difficulties)
        
        # Weighted scores by difficulty
        weighted_scores = scores / (difficulties + 1e-8)
        
        # Compute quantile
        n = len(weighted_scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(weighted_scores, q)
        self.score_function = lambda s, d: s / (d + 1e-8)
    
    def predict(self, data, features=None) -> ConformalPrediction:
        """Adaptive conformal prediction."""
        if self.quantile is None:
            raise ValueError("Must calibrate first")
        
        # Get point prediction
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                point = self.model(data).item()
        else:
            point = self.model.predict(data)
        
        # Estimate difficulty for this sample
        if self.difficulty_estimator:
            difficulty = self.difficulty_estimator(data, features)
        else:
            difficulty = 1.0
        
        # Adaptive interval width
        adaptive_quantile = self.quantile * (difficulty + 1e-8)
        
        lower = point - adaptive_quantile
        upper = point + adaptive_quantile
        width = 2 * adaptive_quantile
        
        return ConformalPrediction(
            point_estimate=point,
            prediction_interval=(lower, upper),
            coverage=self.coverage,
            interval_width=width
        )


def evaluate_coverage(
    predictions: List[ConformalPrediction],
    true_values: List[float]
) -> dict:
    """Evaluate conformal prediction coverage.
    
    Args:
        predictions: List of conformal predictions
        true_values: True target values
    
    Returns:
        Dictionary with coverage metrics
    """
    n = len(predictions)
    covered = 0
    widths = []
    
    for pred, true_val in zip(predictions, true_values):
        lower, upper = pred.prediction_interval
        
        if lower <= true_val <= upper:
            covered += 1
        
        widths.append(pred.interval_width)
    
    empirical_coverage = covered / n
    mean_width = np.mean(widths)
    median_width = np.median(widths)
    
    return {
        'empirical_coverage': empirical_coverage,
        'target_coverage': predictions[0].coverage,
        'coverage_gap': abs(empirical_coverage - predictions[0].coverage),
        'mean_interval_width': mean_width,
        'median_interval_width': median_width,
        'n_samples': n
    }
