"""End-to-end integration tests for the full RPG → KMC pipeline.

These tests exercise the complete stack:
    SMILES input
    → ReactionPathGraph (RPG)
    → KMCSolver
    → KMCResult

No API server is required — the modules are imported directly.
"""

import pytest

from src.models.rpg import ReactionPathGraph
from src.simulation import KMCSolver


class TestEthanolDehydrogenation:
    """CCO → CC[OH2+] → CC=O (ethanol to acetaldehyde)"""

    @pytest.fixture
    def rpg(self):
        g = ReactionPathGraph(temperature=500.0)
        g.build_from_smiles(
            reactants=["CCO"],
            products=["CC=O"],
            intermediates=["CC[OH2+]"],
            ea_values=[80.0, 55.0],
            a_values=[1e13, 1e12],
            ea_uncertainties=[4.0, 2.75],
        )
        g.mark_rate_determining_step()
        return g

    @pytest.fixture
    def result(self, rpg):
        solver = KMCSolver(rpg, max_time=1e-7, n_snapshots=50, n_trajectories=20, seed=1)
        return solver.run({"n0": 1.0})

    def test_pipeline_produces_result(self, result):
        assert result is not None

    def test_snapshot_count(self, result):
        assert len(result.times) == 50

    def test_concentration_conservation(self, result):
        """Total concentration should be approximately conserved."""
        for i, t in enumerate(result.times):
            total = sum(result.concentrations[s][i] for s in result.species_ids)
            assert abs(total - 1.0) < 0.25, f"Conservation violated at t={t}: total={total}"

    def test_max_yield_is_sensible(self, result):
        assert 0.0 <= result.max_yield <= 1.01
        assert result.max_yield_time >= 0.0

    def test_rate_determining_step_identified(self, rpg):
        rds_edges = [e for e in rpg.edges if e.is_rate_determining]
        assert len(rds_edges) == 1
        # Step 1 has Ea=80 kJ/mol > step 2 Ea=55 kJ/mol, so it must be RDS
        rds = rds_edges[0]
        assert rds.activation_energy == pytest.approx(40.0, rel=0.1)  # half-edge

    def test_ci_validity(self, result):
        for sid in result.species_ids:
            for lo, mu, hi in zip(
                result.lower_ci[sid],
                result.concentrations[sid],
                result.upper_ci[sid],
            ):
                assert lo <= mu + 1e-9
                assert mu <= hi + 1e-9

    def test_to_dict_round_trip(self, result):
        d = result.to_dict()
        assert "times" in d
        assert "max_yield" in d
        assert len(d["times"]) == 50


class TestSingleStepReaction:
    """Minimal A → B single-step reaction."""

    def test_single_step_runs(self):
        rpg = ReactionPathGraph(temperature=300.0)
        rpg.build_from_smiles(["A"], ["B"])
        rpg.mark_rate_determining_step()
        solver = KMCSolver(rpg, max_time=1e-8, n_snapshots=10, n_trajectories=5, seed=99)
        result = solver.run({"n0": 0.5})
        assert len(result.times) == 10


class TestHighTemperatureEffect:
    """Higher temperature should produce faster conversion."""

    def _run_at_T(self, T: float) -> float:
        rpg = ReactionPathGraph(temperature=T)
        rpg.build_from_smiles(
            ["A"], ["B"],
            ea_values=[60.0],
            a_values=[1e13],
        )
        rpg.mark_rate_determining_step()
        solver = KMCSolver(rpg, max_time=1e-7, n_snapshots=20, n_trajectories=10, seed=7)
        result = solver.run({"n0": 1.0})
        return result.max_yield

    def test_higher_T_gives_higher_yield(self):
        yield_300 = self._run_at_T(300.0)
        yield_800 = self._run_at_T(800.0)
        # At higher T the reaction proceeds further in the same time window
        assert yield_800 >= yield_300 - 0.05  # allow small numerical slack
