"""Tests for the KMC solver and Arrhenius utilities."""

import math
import pytest

from src.models.rpg import ReactionPathGraph
from src.simulation import KMCSolver, arrhenius_rate, activation_energy_from_rates


# ---------------------------------------------------------------------------
# Arrhenius utilities
# ---------------------------------------------------------------------------


class TestArrheniusUtils:
    def test_basic_rate(self):
        k = arrhenius_rate(frequency_factor=1e13, activation_energy=80.0, temperature=500.0)
        assert k > 0
        # Higher temperature should give higher rate
        k_hot = arrhenius_rate(1e13, 80.0, 700.0)
        assert k_hot > k

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            arrhenius_rate(1e13, 80.0, 0.0)

    def test_activation_energy_back_calculation(self):
        A, Ea = 1e13, 80.0
        k1 = arrhenius_rate(A, Ea, 500.0)
        k2 = arrhenius_rate(A, Ea, 700.0)
        Ea_calc = activation_energy_from_rates(k1, 500.0, k2, 700.0)
        assert abs(Ea_calc - Ea) < 0.01  # within 0.01 kJ/mol


# ---------------------------------------------------------------------------
# ReactionPathGraph
# ---------------------------------------------------------------------------


class TestReactionPathGraph:
    def _build_simple_rpg(self):
        rpg = ReactionPathGraph(temperature=500.0)
        rpg.build_from_smiles(
            reactants=["CCO"],
            products=["CC=O"],
            intermediates=["CC[OH2+]"],
            ea_values=[80.0, 60.0],
            a_values=[1e13, 1e13],
        )
        return rpg

    def test_node_count(self):
        rpg = self._build_simple_rpg()
        # reactant + intermediate + TS1 + TS2 + product = 5 nodes
        assert len(rpg.nodes) == 5

    def test_edge_count(self):
        rpg = self._build_simple_rpg()
        # Each step is split into 2 half-edges via TS → 2 steps × 2 = 4 edges
        assert len(rpg.edges) == 4

    def test_rate_determining_step_marked(self):
        rpg = self._build_simple_rpg()
        rpg.mark_rate_determining_step()
        rds_edges = [e for e in rpg.edges if e.is_rate_determining]
        assert len(rds_edges) == 1

    def test_to_dict_structure(self):
        rpg = self._build_simple_rpg()
        d = rpg.to_dict()
        assert "nodes" in d and "edges" in d
        for node in d["nodes"]:
            assert "id" in node and "type" in node and "gibbs_free_energy" in node
        for edge in d["edges"]:
            assert "activation_energy" in edge and "rate_constant_at_T" in edge

    def test_single_step_reaction(self):
        rpg = ReactionPathGraph(temperature=298.15)
        rpg.build_from_smiles(["A"], ["B"])
        assert len(rpg.nodes) >= 2


# ---------------------------------------------------------------------------
# KMC Solver
# ---------------------------------------------------------------------------


class TestKMCSolver:
    def _build_rpg_and_solver(self, n_traj=10):
        rpg = ReactionPathGraph(temperature=500.0)
        rpg.build_from_smiles(
            reactants=["CCO"],
            products=["CC=O"],
            intermediates=["CC[OH2+]"],
            ea_values=[80.0, 60.0],
            a_values=[1e13, 1e13],
        )
        rpg.mark_rate_determining_step()
        solver = KMCSolver(
            rpg, max_time=1e-6, n_snapshots=20, n_trajectories=n_traj, seed=0
        )
        return solver

    def test_result_has_correct_snapshot_count(self):
        solver = self._build_rpg_and_solver()
        result = solver.run({"n0": 1.0})
        assert len(result.times) == 20

    def test_concentration_sums_are_non_negative(self):
        solver = self._build_rpg_and_solver()
        result = solver.run({"n0": 1.0})
        for species_concs in result.concentrations.values():
            assert all(c >= 0 for c in species_concs)

    def test_ci_lower_le_mean_le_upper(self):
        solver = self._build_rpg_and_solver(n_traj=20)
        result = solver.run({"n0": 1.0})
        for sid in result.species_ids:
            for lo, mu, hi in zip(
                result.lower_ci[sid],
                result.concentrations[sid],
                result.upper_ci[sid],
            ):
                assert lo <= mu + 1e-9
                assert mu <= hi + 1e-9

    def test_max_yield_between_0_and_1(self):
        solver = self._build_rpg_and_solver()
        result = solver.run({"n0": 1.0})
        assert 0.0 <= result.max_yield <= 1.0 + 1e-6

    def test_to_dict_keys(self):
        solver = self._build_rpg_and_solver()
        result = solver.run({"n0": 1.0})
        d = result.to_dict()
        for key in ["times", "concentrations", "lower_ci", "upper_ci",
                    "max_yield_time", "max_yield", "n_trajectories"]:
            assert key in d
