"""
Phase 4: Scientific Validity & Data Infrastructure Demo.

This script demonstrates:
1. Dataset validation with scientific constraints
2. Monte Carlo ensemble execution with uncertainty quantification
3. Benchmark dataset registry and schema management
4. Explicit preprocessing DAG execution

Purpose:
- Show that the framework now has scientific validity foundations
- Demonstrate reproducible stochastic experiments
- Show dataset governance infrastructure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemdynamics.data import (
    ReactionDataset,
    ChemicalReaction,
    ReactionConditions,
    ReactionValidator,
    DatasetRegistry,
    DatasetMetadata,
    DatasetType,
    PreprocessingDAG,
    PreprocessingStage,
    PreprocessingStageType,
    create_basic_preprocessing_pipeline
)
from chemdynamics.simulation import (
    MonteCarloEnsemble,
    StochasticSimulationEngine
)
from chemdynamics.config.schema import SimulationConfig


def demo_dataset_validation():
    """Demonstrate scientific dataset validation."""
    print("\n" + "=" * 80)
    print("1. DATASET VALIDATION DEMO")
    print("=" * 80)

    dataset = ReactionDataset()

    valid_conditions = ReactionConditions(
        temperature=25.0,
        pressure=1.0,
        ph=7.0,
        time=3600.0
    )

    valid_reaction = ChemicalReaction(
        reaction_id="valid_001",
        reactants=["CCO", "CC(=O)O"],
        products=["CCOC(=O)C"],
        conditions=valid_conditions,
        reaction_rate=0.001,
        yield_percentage=85.0
    )

    invalid_temp_reaction = ChemicalReaction(
        reaction_id="invalid_temp",
        reactants=["C"],
        products=["CO2"],
        conditions=ReactionConditions(temperature=-300.0),
        reaction_rate=0.001
    )

    invalid_rate_reaction = ChemicalReaction(
        reaction_id="invalid_rate",
        reactants=["H2", "O2"],
        products=["H2O"],
        conditions=ReactionConditions(temperature=25.0),
        reaction_rate=-0.001
    )

    invalid_yield_reaction = ChemicalReaction(
        reaction_id="invalid_yield",
        reactants=["N2", "H2"],
        products=["NH3"],
        conditions=ReactionConditions(temperature=400.0),
        reaction_rate=0.0005,
        yield_percentage=120.0
    )

    dataset.add_reaction(valid_reaction)
    dataset.add_reaction(invalid_temp_reaction)
    dataset.add_reaction(invalid_rate_reaction)
    dataset.add_reaction(invalid_yield_reaction)

    validator = ReactionValidator()
    report = validator.validate_dataset(dataset.reactions)

    print(f"\nTotal reactions: {report.total_reactions}")
    print(f"Valid reactions: {report.valid_reactions}")
    print(f"Error count: {report.error_count}")
    print(f"Warning count: {report.warning_count}")
    print(f"Is scientifically valid: {report.is_valid}")

    print("\nValidation Issues:")
    for issue in report.issues:
        print(f"  - [{issue.severity.value}] {issue.category.value}: {issue.message}")


def demo_monte_carlo_ensemble():
    """Demonstrate Monte Carlo ensemble execution."""
    print("\n" + "=" * 80)
    print("2. MONTE CARLO ENSEMBLE DEMO")
    print("=" * 80)

    config = SimulationConfig(
        num_steps=50,
        dt=0.1,
        max_concentration=1000.0,
        allow_negative_concentration=False,
        noise_scale=0.05
    )

    initial_concentrations = {
        "A": 1.0,
        "B": 0.5
    }

    print(f"\nRunning ensemble with {config.num_steps} steps...")
    ensemble = MonteCarloEnsemble(config, base_seed=42)
    trajectories = ensemble.run_ensemble(
        initial_concentrations,
        num_rollouts=20
    )

    print(f"Generated {len(trajectories)} trajectories")

    summary = ensemble.compute_summary()
    print(f"\nEnsemble Summary:")
    print(f"  Rollouts: {summary.num_rollouts}")
    print(f"  Steps: {summary.num_steps}")
    print(f"  Species: {summary.species}")

    for species, stats in summary.trajectory_stats.items():
        final_mean = stats.mean[-1]
        final_std = stats.std[-1]
        final_ci_low = stats.ci_lower[-1]
        final_ci_high = stats.ci_upper[-1]
        print(f"\n  {species}:")
        print(f"    Final mean: {final_mean:.4f}")
        print(f"    Final std: {final_std:.4f}")
        print(f"    95% CI: [{final_ci_low:.4f}, {final_ci_high:.4f}]")

    is_reproducible = ensemble.verify_reproducibility(initial_concentrations)
    print(f"\nReproducibility check: {'PASSED' if is_reproducible else 'FAILED'}")

    anomalies = ensemble.detect_anomalies()
    print(f"Anomalies detected: {anomalies.num_anomalies}")


def demo_dataset_registry():
    """Demonstrate benchmark dataset registry."""
    print("\n" + "=" * 80)
    print("3. DATASET REGISTRY DEMO")
    print("=" * 80)

    registry = DatasetRegistry(Path("data/demo_registry"))

    metadata = DatasetMetadata(
        name="demo_benchmark",
        version="1.0.0",
        dataset_type=DatasetType.BENCHMARK,
        description="Demo benchmark dataset for validation",
        schema_version="1.0.0",
        num_reactions=100,
        created_at="2026-05-07T00:00:00",
        source="Demo",
        tags=["demo", "benchmark", "test"]
    )

    registry.register_dataset(metadata)
    print(f"\nRegistered dataset: {metadata.name} v{metadata.version}")

    datasets = registry.list_datasets()
    print(f"Available datasets: {datasets}")

    versions = registry.list_versions("demo_benchmark")
    print(f"Versions for demo_benchmark: {versions}")

    retrieved = registry.get_dataset("demo_benchmark")
    print(f"\nRetrieved metadata: {retrieved.name}")


def demo_preprocessing_dag():
    """Demonstrate explicit preprocessing DAG."""
    print("\n" + "=" * 80)
    print("4. PREPROCESSING DAG DEMO")
    print("=" * 80)

    dataset = ReactionDataset()

    for i in range(10):
        temp = 25.0 if i % 2 == 0 else -300.0
        rate = 0.001 * (i + 1) if i % 3 == 0 else None

        reaction = ChemicalReaction(
            reaction_id=f"rxn_{i:03d}",
            reactants=["C"],
            products=["CO2"],
            conditions=ReactionConditions(temperature=temp),
            reaction_rate=rate
        )
        dataset.add_reaction(reaction)

    print(f"\nInput reactions: {len(dataset.reactions)}")

    dag = create_basic_preprocessing_pipeline()
    processed = dag.execute(dataset.reactions)

    print(f"Processed reactions: {len(processed)}")

    print("\nExecution Logs:")
    for log in dag.execution_logs:
        status = "SUCCESS" if log.success else "FAILED"
        print(f"  {log.stage_name}: {status} ({log.duration_seconds:.4f}s)")


def main():
    print("\n" + "=" * 80)
    print("CHEMDYNAMICS PHASE 4: SCIENTIFIC VALIDITY & DATA INFRASTRUCTURE")
    print("=" * 80)

    try:
        demo_dataset_validation()
        demo_monte_carlo_ensemble()
        demo_dataset_registry()
        demo_preprocessing_dag()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey scientific validity improvements:")
        print("  1. ✅ Dataset validation with physical constraints")
        print("  2. ✅ Monte Carlo ensemble with uncertainty quantification")
        print("  3. ✅ Benchmark dataset registry & versioning")
        print("  4. ✅ Explicit preprocessing DAG with provenance")
        print("  5. ✅ Reproducible stochastic simulations")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
