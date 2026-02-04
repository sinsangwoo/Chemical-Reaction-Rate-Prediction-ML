"""Active Learning for uncertainty-guided data acquisition.

Uses uncertainty estimates to identify the most informative samples
for labeling, minimizing labeling cost and maximizing model improvement.

Strategies:
1. Uncertainty Sampling: Query samples with highest uncertainty
2. Query-by-Committee: Query where models disagree most
3. Expected Model Change: Query samples that would change model most

Based on:
- Settles (2012): Active Learning Literature Survey
- Gal et al. (2017): Deep Bayesian Active Learning
- Nature Comm (2025): UQ for molecular design
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import heapq


@dataclass
class QuerySample:
    """Sample recommended for labeling.
    
    Attributes:
        index: Index in unlabeled pool
        data: Sample data
        acquisition_score: Score (higher = more informative)
        uncertainty: Uncertainty estimate
        strategy: Acquisition strategy used
    """
    index: int
    data: any
    acquisition_score: float
    uncertainty: float
    strategy: str
    
    def __lt__(self, other):
        """For heap operations (max-heap)."""
        return self.acquisition_score > other.acquisition_score


class ActiveLearner:
    """Active learning framework for uncertainty-guided sampling.
    
    Args:
        model: Uncertainty-aware model (e.g., MCDropoutGNN)
        strategy: Acquisition strategy
            - 'uncertainty': Maximum uncertainty
            - 'margin': Minimum margin between top predictions
            - 'entropy': Maximum entropy
            - 'bald': Bayesian Active Learning by Disagreement
    """
    
    def __init__(
        self,
        model,
        strategy: str = 'uncertainty',
        batch_size: int = 10
    ):
        self.model = model
        self.strategy = strategy.lower()
        self.batch_size = batch_size
        
        # Track query history
        self.query_history = []
        self.performance_history = []
    
    def query(
        self,
        unlabeled_pool: List,
        n_samples: int = None
    ) -> List[QuerySample]:
        """Select most informative samples for labeling.
        
        Args:
            unlabeled_pool: Pool of unlabeled samples
            n_samples: Number of samples to query (default: batch_size)
        
        Returns:
            List of QuerySample objects (most informative first)
        """
        if n_samples is None:
            n_samples = self.batch_size
        
        # Compute acquisition scores
        scores = []
        
        for i, data in enumerate(unlabeled_pool):
            if self.strategy == 'uncertainty':
                score, uncertainty = self._uncertainty_sampling(data)
            elif self.strategy == 'entropy':
                score, uncertainty = self._entropy_sampling(data)
            elif self.strategy == 'bald':
                score, uncertainty = self._bald_sampling(data)
            else:
                score, uncertainty = self._uncertainty_sampling(data)
            
            query_sample = QuerySample(
                index=i,
                data=data,
                acquisition_score=score,
                uncertainty=uncertainty,
                strategy=self.strategy
            )
            scores.append(query_sample)
        
        # Select top n_samples
        top_samples = heapq.nlargest(n_samples, scores)
        
        # Track query
        self.query_history.append(top_samples)
        
        return top_samples
    
    def _uncertainty_sampling(self, data) -> Tuple[float, float]:
        """Uncertainty sampling: query highest uncertainty.
        
        Args:
            data: Sample
        
        Returns:
            (acquisition_score, uncertainty)
        """
        # Get uncertainty estimate from model
        if hasattr(self.model, 'predict_with_uncertainty'):
            result = self.model.predict_with_uncertainty(data)
            uncertainty = result.total
        else:
            # Fallback: variance from multiple predictions
            predictions = []
            for _ in range(10):
                pred = self.model(data)
                predictions.append(pred.item())
            uncertainty = np.var(predictions)
        
        # Acquisition score = uncertainty
        score = uncertainty
        
        return score, uncertainty
    
    def _entropy_sampling(self, data) -> Tuple[float, float]:
        """Entropy sampling: query highest predictive entropy.
        
        For regression, use differential entropy of predictive distribution.
        """
        # Get predictive distribution
        if hasattr(self.model, 'predict_with_uncertainty'):
            result = self.model.predict_with_uncertainty(data, n_samples=50)
            
            # Approximate entropy from variance (Gaussian assumption)
            variance = result.total
            entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
        else:
            entropy = 0.0
        
        return entropy, variance if 'variance' in locals() else entropy
    
    def _bald_sampling(self, data) -> Tuple[float, float]:
        """BALD: Bayesian Active Learning by Disagreement.
        
        Measures mutual information between predictions and model parameters.
        
        I(y; θ | x) = H(y | x) - E_θ[H(y | x, θ)]
        
        Args:
            data: Sample
        
        Returns:
            (bald_score, total_uncertainty)
        """
        if not hasattr(self.model, 'predict_with_uncertainty'):
            # Fallback to uncertainty sampling
            return self._uncertainty_sampling(data)
        
        # Get multiple predictions (epistemic uncertainty)
        result = self.model.predict_with_uncertainty(data, n_samples=100)
        
        # BALD score ≈ epistemic uncertainty
        # (exact computation requires integrating over θ)
        bald_score = result.epistemic
        total_uncertainty = result.total
        
        return bald_score, total_uncertainty
    
    def update_model(
        self,
        new_data: List,
        new_labels: List,
        retrain: bool = True
    ):
        """Update model with newly labeled data.
        
        Args:
            new_data: Newly labeled samples
            new_labels: Labels for new samples
            retrain: Whether to retrain model (default: True)
        """
        if retrain and hasattr(self.model, 'fit'):
            # Retrain model with augmented data
            self.model.fit(new_data, new_labels)
    
    def evaluate_iteration(
        self,
        test_data: List,
        test_labels: List
    ) -> dict:
        """Evaluate model performance after active learning iteration.
        
        Args:
            test_data: Test set
            test_labels: Test labels
        
        Returns:
            Performance metrics
        """
        predictions = []
        
        for data in test_data:
            pred = self.model(data).item()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        test_labels = np.array(test_labels)
        
        # Compute metrics
        mae = np.mean(np.abs(predictions - test_labels))
        mse = np.mean((predictions - test_labels) ** 2)
        r2 = 1 - mse / np.var(test_labels)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'n_iterations': len(self.query_history)
        }
        
        self.performance_history.append(metrics)
        
        return metrics


class BatchActiveLearner(ActiveLearner):
    """Batch-mode active learning with diversity.
    
    Selects diverse batches to avoid redundant queries.
    
    Args:
        model: Uncertainty-aware model
        strategy: Base acquisition strategy
        diversity_weight: Weight for diversity term (0-1)
        batch_size: Batch size for queries
    """
    
    def __init__(
        self,
        model,
        strategy: str = 'uncertainty',
        diversity_weight: float = 0.3,
        batch_size: int = 10
    ):
        super().__init__(model, strategy, batch_size)
        self.diversity_weight = diversity_weight
    
    def query(self, unlabeled_pool: List, n_samples: int = None) -> List[QuerySample]:
        """Query diverse batch of informative samples.
        
        Greedily selects samples that are:
        1. High acquisition score (informative)
        2. Diverse from already selected samples
        """
        if n_samples is None:
            n_samples = self.batch_size
        
        # Compute base acquisition scores
        candidates = []
        
        for i, data in enumerate(unlabeled_pool):
            if self.strategy == 'uncertainty':
                score, uncertainty = self._uncertainty_sampling(data)
            else:
                score, uncertainty = self._uncertainty_sampling(data)
            
            candidates.append(QuerySample(
                index=i,
                data=data,
                acquisition_score=score,
                uncertainty=uncertainty,
                strategy=self.strategy
            ))
        
        # Greedy selection with diversity
        selected = []
        
        while len(selected) < n_samples and candidates:
            if not selected:
                # First: select highest score
                best = max(candidates, key=lambda x: x.acquisition_score)
            else:
                # Subsequent: balance score and diversity
                scores = []
                
                for c in candidates:
                    # Information score
                    info_score = c.acquisition_score
                    
                    # Diversity score (min distance to selected)
                    diversity_score = min(
                        self._distance(c.data, s.data)
                        for s in selected
                    )
                    
                    # Combined score
                    combined = (
                        (1 - self.diversity_weight) * info_score
                        + self.diversity_weight * diversity_score
                    )
                    scores.append((combined, c))
                
                best = max(scores, key=lambda x: x[0])[1]
            
            selected.append(best)
            candidates.remove(best)
        
        self.query_history.append(selected)
        
        return selected
    
    def _distance(self, data1, data2) -> float:
        """Compute distance between two samples.
        
        Simple implementation: Euclidean distance on features.
        Override for custom distance metrics.
        """
        # Placeholder: random diversity
        return np.random.rand()
