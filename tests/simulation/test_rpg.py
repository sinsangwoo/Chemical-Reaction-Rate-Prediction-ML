"""Tests for the Reaction Path Graph module."""

import pytest

from src.models.rpg import (
    ReactionPathGraph,
    ElementaryReaction,
    ReactionNode,
    NodeType,
)


class TestNodeType:
    def test_enum_values(self):
        assert NodeType.REACTANT == "reactant"
        assert NodeType.TRANSITION_STATE == "transition_state"


class TestReactionNode:
    def test_default_label(self):
        node = ReactionNode(
            node_id="n0", smiles="CCO", node_type=NodeType.REACTANT
        )
        assert node.label == "CCO"

    def test_custom_label(self):
        node = ReactionNode(
            node_id="n0", smiles="CCO", node_type=NodeType.REACTANT, label="ethanol"
        )
        assert node.label == "ethanol"


class TestElementaryReaction:
    def test_rate_constant_increases_with_temperature(self):
        rxn = ElementaryReaction(
            reaction_id="r0",
            from_node="n0",
            to_node="n1",
            activation_energy=80.0,
            frequency_factor=1e13,
        )
        k_low = rxn.rate_constant(300.0)
        k_high = rxn.rate_constant(600.0)
        assert k_high > k_low


class TestReactionPathGraphEdgeCases:
    def test_multi_product_graph(self):
        rpg = ReactionPathGraph(temperature=400.0)
        rpg.build_from_smiles(
            reactants=["C", "O"],
            products=["CO"],
        )
        assert len(rpg.nodes) >= 2

    def test_explicit_ea_values(self):
        rpg = ReactionPathGraph(temperature=500.0)
        rpg.build_from_smiles(
            reactants=["A"], products=["C"],
            intermediates=["B"],
            ea_values=[50.0, 30.0],
            a_values=[1e12, 1e12],
        )
        # All edges should have positive Ea
        for edge in rpg.edges:
            assert edge.activation_energy > 0

    def test_mark_rds_idempotent(self):
        rpg = ReactionPathGraph(temperature=500.0)
        rpg.build_from_smiles(["A"], ["B"])
        rds1 = rpg.mark_rate_determining_step()
        rds2 = rpg.mark_rate_determining_step()
        assert rds1 == rds2

    def test_summary_contains_temperature(self):
        rpg = ReactionPathGraph(temperature=750.0)
        rpg.build_from_smiles(["A"], ["B"])
        rpg.mark_rate_determining_step()
        assert "750" in rpg.summary()
