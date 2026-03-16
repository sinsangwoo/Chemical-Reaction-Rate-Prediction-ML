"""Data classes for elementary reactions in a reaction network."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class NodeType(str, Enum):
    """Types of nodes in the Reaction Path Graph."""
    REACTANT = "reactant"
    INTERMEDIATE = "intermediate"
    TRANSITION_STATE = "transition_state"
    PRODUCT = "product"


@dataclass
class ReactionNode:
    """A single species node in the Reaction Path Graph."""
    node_id: str
    smiles: str
    node_type: NodeType
    gibbs_free_energy: float = 0.0   # kJ/mol, relative to reactants
    uncertainty: float = 0.0         # kJ/mol, 1-sigma Bayesian uncertainty
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.smiles[:20]


@dataclass
class ElementaryReaction:
    """A directed edge in the RPG, representing one elementary step.

    Attributes
    ----------
    reaction_id : str
        Unique identifier, e.g. ``"r0"``.
    from_node : str
        ``node_id`` of the source species.
    to_node : str
        ``node_id`` of the target species.
    activation_energy : float
        Forward activation energy Ea (kJ/mol).
    frequency_factor : float
        Pre-exponential factor A (s^-1 or L/mol/s for bimolecular).
    ea_uncertainty : float
        1-sigma uncertainty on Ea from Bayesian GNN (kJ/mol).
    a_uncertainty : float
        1-sigma uncertainty on A.
    is_rate_determining : bool
        Set to True for the step with the highest Ea in the network.
    """

    reaction_id: str
    from_node: str
    to_node: str
    activation_energy: float         # kJ/mol
    frequency_factor: float          # s^-1
    ea_uncertainty: float = 0.0
    a_uncertainty: float = 0.0
    is_rate_determining: bool = False
    stoichiometry: float = 1.0

    def rate_constant(self, temperature: float) -> float:
        """Compute k(T) via the Arrhenius equation.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Rate constant k  (same units as ``frequency_factor``).
        """
        R = 8.314e-3  # kJ / (mol · K)
        import math
        return self.frequency_factor * math.exp(-self.activation_energy / (R * temperature))
