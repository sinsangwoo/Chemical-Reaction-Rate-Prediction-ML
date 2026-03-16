"""Reaction Path Graph (RPG) builder.

Constructs a directed graph of elementary reactions from a list of
SMILES strings.  In production this module calls the GNN backend
(``HybridGNN``) to predict Ea and A for each edge; the current
implementation provides a deterministic fallback so the rest of the
pipeline can run without a trained model checkpoint.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

from .elementary_reaction import ElementaryReaction, NodeType, ReactionNode


class ReactionPathGraph:
    """Directed graph representing a multi-step reaction network.

    Parameters
    ----------
    temperature : float
        Operating temperature in Kelvin (used for rate-constant display).

    Examples
    --------
    >>> rpg = ReactionPathGraph(temperature=500.0)
    >>> rpg.build_from_smiles(["CCO"], ["CC=O"], intermediates=["CC[OH2+]"])
    >>> rpg.mark_rate_determining_step()
    >>> print(rpg.summary())
    """

    def __init__(self, temperature: float = 298.15) -> None:
        self.temperature = temperature
        self.nodes: Dict[str, ReactionNode] = {}
        self.edges: List[ElementaryReaction] = []
        self._node_counter = 0
        self._edge_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_smiles(
        self,
        reactants: List[str],
        products: List[str],
        intermediates: Optional[List[str]] = None,
        transition_states: Optional[List[str]] = None,
        ea_values: Optional[List[float]] = None,
        a_values: Optional[List[float]] = None,
        ea_uncertainties: Optional[List[float]] = None,
    ) -> "ReactionPathGraph":
        """Populate the graph from SMILES lists.

        Parameters
        ----------
        reactants, products : list of str
            SMILES for reactant / product species.
        intermediates : list of str, optional
            SMILES for intermediate species (auto-generated if omitted).
        transition_states : list of str, optional
            Labels for transition states (auto-generated if omitted).
        ea_values : list of float, optional
            Activation energies (kJ/mol) for each step. Length must equal
            ``len(intermediates) + 1`` when provided.
        a_values : list of float, optional
            Pre-exponential factors (s^-1) for each step.
        ea_uncertainties : list of float, optional
            1-sigma Bayesian uncertainties on each Ea.

        Returns
        -------
        ReactionPathGraph
            *self* (for method chaining).
        """
        intermediates = intermediates or []
        transition_states = transition_states or []

        all_species: List[Tuple[str, NodeType]] = (
            [(s, NodeType.REACTANT) for s in reactants]
            + [(s, NodeType.INTERMEDIATE) for s in intermediates]
            + [(s, NodeType.PRODUCT) for s in products]
        )

        # Build ordered chain: reactant(s) → intermediate(s) → product(s)
        species_ids: List[str] = []
        for smiles, ntype in all_species:
            nid = self._add_node(smiles, ntype)
            species_ids.append(nid)

        n_steps = len(species_ids) - 1
        ea_vals = ea_values or self._default_ea(n_steps)
        a_vals = a_values or [1e13] * n_steps
        ea_unc = ea_uncertainties or [ea * 0.05 for ea in ea_vals]

        ts_labels = list(transition_states)
        for i, (ts_label, ea, a, dea) in enumerate(
            zip(
                itertools.chain(ts_labels, itertools.repeat(None)),
                ea_vals,
                a_vals,
                ea_unc,
            )
        ):
            if i >= n_steps:
                break
            from_id = species_ids[i]
            to_id = species_ids[i + 1]

            if ts_label is None:
                ts_label = f"TS{i + 1}"

            # Insert a transition-state node between each pair
            ts_g = self._add_gibbs_for_ts(
                from_nid=from_id, ea=ea, ts_label=ts_label
            )
            ts_nid = self._add_node(ts_label, NodeType.TRANSITION_STATE,
                                    gibbs=ts_g, label=ts_label)

            self._add_edge(from_id, ts_nid, ea=ea / 2, a=a,
                           ea_unc=dea / 2, reaction_id=f"r{self._edge_counter}")
            self._add_edge(ts_nid, to_id, ea=ea / 2, a=a,
                           ea_unc=dea / 2, reaction_id=f"r{self._edge_counter}")

        # Set relative Gibbs energies for the product chain
        self._assign_gibbs_chain(species_ids)
        return self

    def mark_rate_determining_step(self) -> Optional[str]:
        """Identify and flag the step with the highest net Ea.

        Returns
        -------
        str or None
            ``reaction_id`` of the rate-determining step.
        """
        if not self.edges:
            return None
        rds = max(self.edges, key=lambda e: e.activation_energy)
        for edge in self.edges:
            edge.is_rate_determining = edge.reaction_id == rds.reaction_id
        return rds.reaction_id

    def to_dict(self) -> dict:
        """Serialise the graph to a JSON-compatible dict."""
        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "smiles": n.smiles,
                    "type": n.node_type.value,
                    "gibbs_free_energy": round(n.gibbs_free_energy, 3),
                    "uncertainty": round(n.uncertainty, 3),
                    "label": n.label,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.reaction_id,
                    "from": e.from_node,
                    "to": e.to_node,
                    "activation_energy": round(e.activation_energy, 3),
                    "frequency_factor": e.frequency_factor,
                    "ea_uncertainty": round(e.ea_uncertainty, 3),
                    "is_rate_determining": e.is_rate_determining,
                    "rate_constant_at_T": round(
                        e.rate_constant(self.temperature), 6
                    ),
                }
                for e in self.edges
            ],
            "temperature": self.temperature,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        rds = next((e for e in self.edges if e.is_rate_determining), None)
        lines = [
            f"ReactionPathGraph  T={self.temperature} K",
            f"  nodes : {len(self.nodes)}",
            f"  edges : {len(self.edges)}",
        ]
        if rds:
            lines.append(
                f"  rate-determining step : {rds.reaction_id}  "
                f"Ea={rds.activation_energy:.1f} kJ/mol"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_node(
        self,
        smiles: str,
        node_type: NodeType,
        gibbs: float = 0.0,
        label: str = "",
    ) -> str:
        nid = f"n{self._node_counter}"
        self._node_counter += 1
        self.nodes[nid] = ReactionNode(
            node_id=nid,
            smiles=smiles,
            node_type=node_type,
            gibbs_free_energy=gibbs,
            label=label or smiles[:20],
        )
        return nid

    def _add_edge(
        self,
        from_id: str,
        to_id: str,
        ea: float,
        a: float,
        ea_unc: float,
        reaction_id: str,
    ) -> None:
        self.edges.append(
            ElementaryReaction(
                reaction_id=reaction_id,
                from_node=from_id,
                to_node=to_id,
                activation_energy=ea,
                frequency_factor=a,
                ea_uncertainty=ea_unc,
            )
        )
        self._edge_counter += 1

    def _add_gibbs_for_ts(
        self, from_nid: str, ea: float, ts_label: str
    ) -> float:
        """Return the Gibbs energy of a TS node (reactant G + Ea)."""
        base = self.nodes[from_nid].gibbs_free_energy
        return base + ea

    @staticmethod
    def _default_ea(n_steps: int) -> List[float]:
        """Generate plausible fallback Ea values when GNN is unavailable."""
        base = [80.0, 60.0, 45.0, 30.0, 55.0]
        return [base[i % len(base)] for i in range(n_steps)]

    def _assign_gibbs_chain(
        self, species_ids: List[str], delta_g_per_step: float = -20.0
    ) -> None:
        """Assign monotonically decreasing Gibbs energies along the chain."""
        g = 0.0
        for i, nid in enumerate(species_ids):
            self.nodes[nid].gibbs_free_energy = g
            g += delta_g_per_step
