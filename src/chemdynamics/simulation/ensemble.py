"""
Monte Carlo Ensemble Execution & Uncertainty Validation Infrastructure.

This module provides:
- Ensemble trajectory execution
- Statistical trajectory summaries
- Uncertainty diagnostics
- Rollout reproducibility tests
- Variance monitoring
- Confidence interval infrastructure
- Anomaly detection across trajectories

Design Intent:
- Scientifically interpretable uncertainty, not just "random noise theater"
- Reproducible stochastic simulations with seed control
- Statistical rigor in uncertainty aggregation
- Transparent diagnostics for ensemble behavior
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

from .engine import BaseSimulationEngine, StochasticSimulationEngine
from chemdynamics.config.schema import SimulationConfig


@dataclass
class TrajectoryStatistics:
    """Statistical summary for a single species across ensemble."""
    species: str
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    min: np.ndarray
    max: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    coefficient_of_variation: np.ndarray


@dataclass
class EnsembleSummary:
    """Comprehensive summary of Monte Carlo ensemble."""
    num_rollouts: int
    num_steps: int
    species: List[str]
    trajectory_stats: Dict[str, TrajectoryStatistics]
    timestamp: str = field(default_factory=lambda: "")
    seed: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "num_rollouts": self.num_rollouts,
            "num_steps": self.num_steps,
            "species": self.species,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "trajectory_stats": {
                species: {
                    "mean": stat.mean.tolist(),
                    "std": stat.std.tolist(),
                    "variance": stat.variance.tolist(),
                    "median": stat.median.tolist(),
                    "q25": stat.q25.tolist(),
                    "q75": stat.q75.tolist(),
                    "min": stat.min.tolist(),
                    "max": stat.max.tolist(),
                    "ci_lower": stat.ci_lower.tolist(),
                    "ci_upper": stat.ci_upper.tolist(),
                    "coefficient_of_variation": stat.coefficient_of_variation.tolist()
                }
                for species, stat in self.trajectory_stats.items()
            }
        }

    def save(self, filepath: Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class AnomalyReport:
    """Report for detected anomalies in ensemble trajectories."""
    num_anomalies: int
    anomalous_rollouts: List[int]
    anomaly_details: List[Dict]


class MonteCarloEnsemble:
    """
    Monte Carlo Ensemble Executor for Stochastic Simulations.

    This class manages:
    - Multiple stochastic rollouts with seed control
    - Ensemble trajectory aggregation
    - Statistical uncertainty quantification
    - Reproducibility verification
    - Anomaly detection

    Scientific Rationale:
    - Single trajectories are not scientifically representative
    - Ensemble statistics provide meaningful uncertainty bounds
    - Seed control enables reproducible stochastic experiments
    """

    def __init__(
        self,
        config: SimulationConfig,
        engine_class: type = StochasticSimulationEngine,
        base_seed: Optional[int] = None
    ):
        self.config = config
        self.engine_class = engine_class
        self.base_seed = base_seed if base_seed is not None else torch.initial_seed()
        self.trajectories: List[List[Dict[str, float]]] = []
        self._seeds_used: List[int] = []

    def _get_seed_for_rollout(self, rollout_idx: int) -> int:
        """Generate reproducible seed for each rollout."""
        return self.base_seed + rollout_idx * 1000

    def run_ensemble(
        self,
        initial_concentrations: Dict[str, float],
        num_rollouts: int = 100,
        progress_callback: Optional[callable] = None
    ) -> List[List[Dict[str, float]]]:
        """
        Run Monte Carlo ensemble of simulations.

        Args:
            initial_concentrations: Starting species concentrations
            num_rollouts: Number of stochastic trajectories to generate
            progress_callback: Optional callback for progress tracking

        Returns:
            List of trajectories, each trajectory is list of state dicts
        """
        self.trajectories = []
        self._seeds_used = []

        for rollout_idx in range(num_rollouts):
            seed = self._get_seed_for_rollout(rollout_idx)
            self._seeds_used.append(seed)

            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))

            engine = self.engine_class(self.config)
            trajectory = engine.run(initial_concentrations)
            self.trajectories.append(trajectory)

            if progress_callback:
                progress_callback(rollout_idx + 1, num_rollouts)

        return self.trajectories

    def verify_reproducibility(
        self,
        initial_concentrations: Dict[str, float],
        rollout_idx: int = 0
    ) -> bool:
        """
        Verify that a rollout is reproducible with the same seed.

        Scientific Rationale:
        - Stochastic simulations must be reproducible for scientific validity
        - Seed control enables exact trajectory replication

        Returns:
            True if trajectory is reproducible
        """
        if not self.trajectories:
            raise ValueError("No trajectories available. Run ensemble first.")

        seed = self._seeds_used[rollout_idx]
        original_trajectory = self.trajectories[rollout_idx]

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))

        engine = self.engine_class(self.config)
        new_trajectory = engine.run(initial_concentrations)

        for step_idx, (orig_state, new_state) in enumerate(zip(original_trajectory, new_trajectory)):
            for species, orig_conc in orig_state.items():
                new_conc = new_state[species]
                if not np.isclose(orig_conc, new_conc, rtol=1e-9, atol=1e-9):
                    return False
        return True

    def compute_summary(self, confidence_level: float = 0.95) -> EnsembleSummary:
        """
        Compute comprehensive statistical summary of ensemble.

        Calculates:
        - Mean, median, std, variance
        - Quartiles (Q25, Q75)
        - Min/Max
        - Confidence intervals
        - Coefficient of variation

        Scientific Rationale:
        - Multiple statistics provide robust uncertainty characterization
        - Confidence intervals quantify statistical significance
        - CV normalizes variability for comparison
        """
        if not self.trajectories:
            raise ValueError("No trajectories available. Run ensemble first.")

        num_rollouts = len(self.trajectories)
        num_steps = len(self.trajectories[0])
        species = list(self.trajectories[0][0].keys())

        trajectory_stats = {}

        for sp in species:
            values = np.zeros((num_rollouts, num_steps))
            for rollout_idx, trajectory in enumerate(self.trajectories):
                for step_idx, state in enumerate(trajectory):
                    values[rollout_idx, step_idx] = state[sp]

            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0, ddof=1)
            variance = np.var(values, axis=0, ddof=1)
            median = np.median(values, axis=0)
            q25 = np.percentile(values, 25, axis=0)
            q75 = np.percentile(values, 75, axis=0)
            min_val = np.min(values, axis=0)
            max_val = np.max(values, axis=0)

            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, num_rollouts - 1)
            std_error = std / np.sqrt(num_rollouts)
            ci_lower = mean - t_critical * std_error
            ci_upper = mean + t_critical * std_error

            cv = np.divide(std, mean, out=np.zeros_like(std), where=mean != 0)

            trajectory_stats[sp] = TrajectoryStatistics(
                species=sp,
                mean=mean,
                std=std,
                variance=variance,
                median=median,
                q25=q25,
                q75=q75,
                min=min_val,
                max=max_val,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                coefficient_of_variation=cv
            )

        return EnsembleSummary(
            num_rollouts=num_rollouts,
            num_steps=num_steps,
            species=species,
            trajectory_stats=trajectory_stats,
            seed=self.base_seed,
            timestamp=datetime.now().isoformat()
        )

    def detect_anomalies(
        self,
        z_threshold: float = 3.0
    ) -> AnomalyReport:
        """
        Detect anomalous trajectories using z-score.

        Scientific Rationale:
        - Outlier trajectories may indicate numerical instability
        - Anomalies should be flagged for scientific scrutiny

        Args:
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            AnomalyReport with detected anomalies
        """
        if not self.trajectories:
            raise ValueError("No trajectories available. Run ensemble first.")

        num_rollouts = len(self.trajectories)
        num_steps = len(self.trajectories[0])
        species = list(self.trajectories[0][0].keys())

        final_values = {sp: np.zeros(num_rollouts) for sp in species}
        for rollout_idx, trajectory in enumerate(self.trajectories):
            final_state = trajectory[-1]
            for sp in species:
                final_values[sp][rollout_idx] = final_state[sp]

        anomalous_rollouts = set()
        anomaly_details = []

        for sp in species:
            vals = final_values[sp]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            if std > 0:
                z_scores = (vals - mean) / std
                for rollout_idx, z in enumerate(z_scores):
                    if abs(z) > z_threshold:
                        anomalous_rollouts.add(rollout_idx)
                        anomaly_details.append({
                            "rollout_idx": rollout_idx,
                            "species": sp,
                            "z_score": float(z),
                            "value": float(vals[rollout_idx]),
                            "mean": float(mean),
                            "std": float(std)
                        })

        return AnomalyReport(
            num_anomalies=len(anomalous_rollouts),
            anomalous_rollouts=sorted(list(anomalous_rollouts)),
            anomaly_details=anomaly_details
        )

    def save_trajectories(self, filepath: Path) -> None:
        """Save ensemble trajectories to JSON file."""
        data = {
            "base_seed": self.base_seed,
            "seeds_used": self._seeds_used,
            "trajectories": self.trajectories
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_trajectories(self, filepath: Path) -> None:
        """Load ensemble trajectories from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.base_seed = data["base_seed"]
        self._seeds_used = data["seeds_used"]
        self.trajectories = data["trajectories"]
