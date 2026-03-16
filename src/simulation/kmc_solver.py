"""Kinetic Monte Carlo (KMC) Solver using the Gillespie algorithm.

Simulates the time evolution of species concentrations in an elementary
reaction network.  Propagation is stochastic; confidence intervals are
derived by running multiple independent trajectories with perturbed
rate constants drawn from the Bayesian uncertainty estimates supplied
by the GNN model.

Usage
-----
    from src.simulation import KMCSolver
    from src.models.rpg import ReactionPathGraph

    rpg = ReactionPathGraph(temperature=500)
    rpg.build_from_smiles(["CCO"], ["CC=O"], intermediates=["CC[OH2+]"])
    rpg.mark_rate_determining_step()

    solver = KMCSolver(rpg, max_time=1.0, n_trajectories=50)
    result = solver.run({"n0": 1.0, "n1": 0.0, "n2": 0.0})
    print(result.to_dict())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .arrhenius_utils import arrhenius_rate


@dataclass
class KMCResult:
    """Container for KMC simulation output.

    Attributes
    ----------
    times : list of float
        Snapshot times (s).
    species_ids : list of str
        Ordered list of node IDs whose concentrations are tracked.
    concentrations : dict[str, list[float]]
        Mean concentration (mol/L) over trajectories at each snapshot.
    lower_ci : dict[str, list[float]]
        Lower bound of the 95 % confidence interval.
    upper_ci : dict[str, list[float]]
        Upper bound of the 95 % confidence interval.
    max_yield_time : float
        Time (s) at which the primary product reaches maximum yield.
    max_yield : float
        Maximum fractional yield of the primary product.
    n_trajectories : int
        Number of Gillespie trajectories averaged.
    """

    times: List[float] = field(default_factory=list)
    species_ids: List[str] = field(default_factory=list)
    concentrations: Dict[str, List[float]] = field(default_factory=dict)
    lower_ci: Dict[str, List[float]] = field(default_factory=dict)
    upper_ci: Dict[str, List[float]] = field(default_factory=dict)
    max_yield_time: float = 0.0
    max_yield: float = 0.0
    n_trajectories: int = 0

    def to_dict(self) -> dict:
        return {
            "times": self.times,
            "species_ids": self.species_ids,
            "concentrations": self.concentrations,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "max_yield_time": round(self.max_yield_time, 6),
            "max_yield": round(self.max_yield, 4),
            "n_trajectories": self.n_trajectories,
        }


class KMCSolver:
    """Gillespie-algorithm KMC solver for elementary reaction networks.

    Parameters
    ----------
    rpg : ReactionPathGraph
        The reaction network (nodes + edges with Ea/A values).
    max_time : float
        Total simulation wall-clock time (seconds).
    n_snapshots : int
        Number of evenly spaced time points to record.
    n_trajectories : int
        Number of independent stochastic trajectories to average.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        rpg,  # ReactionPathGraph – avoid circular import via type hint
        max_time: float = 1.0,
        n_snapshots: int = 200,
        n_trajectories: int = 50,
        seed: Optional[int] = 42,
    ) -> None:
        self.rpg = rpg
        self.max_time = max_time
        self.n_snapshots = n_snapshots
        self.n_trajectories = n_trajectories
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        initial_concentrations: Dict[str, float],
    ) -> KMCResult:
        """Run the KMC simulation.

        Parameters
        ----------
        initial_concentrations : dict
            Mapping of node_id → initial concentration (mol/L).
            Species not listed default to 0.

        Returns
        -------
        KMCResult
        """
        species_ids = list(self.rpg.nodes.keys())
        snapshot_times = [
            self.max_time * i / (self.n_snapshots - 1)
            for i in range(self.n_snapshots)
        ]

        all_trajectories: List[Dict[str, List[float]]] = []

        for _ in range(self.n_trajectories):
            conc_trace = self._single_trajectory(
                species_ids, initial_concentrations, snapshot_times
            )
            all_trajectories.append(conc_trace)

        # Aggregate: mean + 95 % CI (±1.96 σ / √n)
        mean_conc: Dict[str, List[float]] = {s: [] for s in species_ids}
        lower_ci: Dict[str, List[float]] = {s: [] for s in species_ids}
        upper_ci: Dict[str, List[float]] = {s: [] for s in species_ids}

        for t_idx in range(self.n_snapshots):
            for s in species_ids:
                vals = [traj[s][t_idx] for traj in all_trajectories]
                mu = sum(vals) / len(vals)
                variance = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
                sigma = math.sqrt(variance)
                half_width = 1.96 * sigma / math.sqrt(len(vals))
                mean_conc[s].append(round(mu, 6))
                lower_ci[s].append(round(max(mu - half_width, 0.0), 6))
                upper_ci[s].append(round(mu + half_width, 6))

        # Find max-yield time for the last product node
        product_id = species_ids[-1]
        max_yield = max(mean_conc[product_id])
        max_yield_time = snapshot_times[
            mean_conc[product_id].index(max_yield)
        ]
        total_initial = sum(initial_concentrations.get(s, 0.0) for s in species_ids)
        fractional_yield = max_yield / total_initial if total_initial > 0 else 0.0

        return KMCResult(
            times=snapshot_times,
            species_ids=species_ids,
            concentrations=mean_conc,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            max_yield_time=max_yield_time,
            max_yield=fractional_yield,
            n_trajectories=self.n_trajectories,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_rate_constants(
        self, perturb: bool = False
    ) -> Dict[str, float]:
        """Build a dict of reaction_id → k(T), optionally perturbed."""
        rates: Dict[str, float] = {}
        for edge in self.rpg.edges:
            k = arrhenius_rate(
                edge.frequency_factor,
                edge.activation_energy,
                self.rpg.temperature,
            )
            if perturb and edge.ea_uncertainty > 0:
                # Sample Ea from N(Ea, σ²) for Bayesian propagation
                ea_sample = self._rng.gauss(
                    edge.activation_energy, edge.ea_uncertainty
                )
                k = arrhenius_rate(
                    edge.frequency_factor,
                    max(ea_sample, 1.0),  # Ea must stay positive
                    self.rpg.temperature,
                )
            rates[edge.reaction_id] = k
        return rates

    def _single_trajectory(
        self,
        species_ids: List[str],
        initial_concentrations: Dict[str, float],
        snapshot_times: List[float],
    ) -> Dict[str, List[float]]:
        """Run one Gillespie trajectory and return concentration snapshots."""
        conc: Dict[str, float] = {
            s: initial_concentrations.get(s, 0.0) for s in species_ids
        }
        rates = self._get_rate_constants(perturb=True)
        t = 0.0
        snap_idx = 0
        trace: Dict[str, List[float]] = {s: [] for s in species_ids}

        while snap_idx < len(snapshot_times):
            # Record snapshots before the next event
            while snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx]:
                for s in species_ids:
                    trace[s].append(conc[s])
                snap_idx += 1

            if snap_idx >= len(snapshot_times):
                break

            # Build propensity list
            propensities: List[Tuple[str, str, str, float]] = []
            for edge in self.rpg.edges:
                k = rates.get(edge.reaction_id, 0.0)
                c_from = conc.get(edge.from_node, 0.0)
                prop = k * max(c_from, 0.0)
                propensities.append((edge.reaction_id, edge.from_node, edge.to_node, prop))

            total_prop = sum(p[3] for p in propensities)

            if total_prop <= 0:
                # No more reactions possible — fill remaining snapshots
                for s in species_ids:
                    remaining = len(snapshot_times) - snap_idx
                    trace[s].extend([conc[s]] * remaining)
                break

            # Time to next event (exponential distribution)
            dt = -math.log(self._rng.random()) / total_prop
            t += dt

            # Select which reaction fires
            r = self._rng.random() * total_prop
            cumulative = 0.0
            for rxn_id, from_node, to_node, prop in propensities:
                cumulative += prop
                if r <= cumulative:
                    # Fire: transfer concentration
                    delta = min(prop * dt, conc.get(from_node, 0.0))
                    conc[from_node] = max(conc[from_node] - delta, 0.0)
                    conc[to_node] = conc.get(to_node, 0.0) + delta
                    break

        return trace
