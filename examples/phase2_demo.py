#!/usr/bin/env python
"""Phase 2 demonstration: Real chemistry with SMILES and USPTO data.

This script demonstrates the new capabilities:
1. SMILES parsing and validation
2. Molecular feature extraction
3. Reaction dataset management
4. USPTO-style synthetic data generation
5. Feature engineering for ML
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.smiles_parser import SMILESParser, ReactionSMILES
from src.data.reaction_dataset import (
    ReactionDataset,
    ChemicalReaction,
    ReactionConditions,
)
from src.data.uspto_loader import USPTOLoader
from src.features.molecular_features import (
    MolecularFeatureExtractor,
    ReactionFeatureBuilder,
)

from rich.console import Console
from rich.table import Table
import pandas as pd

console = Console()


def demo_smiles_parsing():
    """Demonstrate SMILES parsing capabilities."""
    console.print("\n[bold blue]1. SMILES Parsing Demo[/bold blue]\n")

    parser = SMILESParser()

    # Example molecules
    molecules = {
        "Ethanol": "CCO",
        "Acetic Acid": "CC(=O)O",
        "Benzene": "c1ccccc1",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    }

    table = Table(title="Molecular Analysis")
    table.add_column("Name", style="cyan")
    table.add_column("SMILES", style="green")
    table.add_column("Atoms", justify="right")
    table.add_column("MW (est)", justify="right")
    table.add_column("Aromatic", justify="center")

    for name, smiles in molecules.items():
        features = parser.extract_features(smiles)
        atom_counts = features["atom_counts"]
        total_atoms = sum(atom_counts.values())

        table.add_row(
            name,
            smiles,
            str(total_atoms),
            f"{features['estimated_mw']:.1f}",
            "✓" if features["has_aromatic"] else "✗",
        )

    console.print(table)


def demo_reaction_parsing():
    """Demonstrate reaction SMILES parsing."""
    console.print("\n[bold blue]2. Reaction Parsing Demo[/bold blue]\n")

    reaction_parser = ReactionSMILES()

    # Example reactions
    reactions = [
        ("Esterification", "CCO.CC(=O)O>>CCOC(=O)C"),
        ("Suzuki Coupling", "c1ccccc1Br.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1"),
        ("Nucleophilic Sub", "CCCl>NaOH>CCN"),
    ]

    for name, rxn_smiles in reactions:
        console.print(f"[yellow]{name}:[/yellow]")
        console.print(f"  Reaction: {rxn_smiles}")

        parsed = reaction_parser.parse_reaction(rxn_smiles)
        console.print(f"  Reactants: {len(parsed['reactants'])}")
        console.print(f"  Products: {len(parsed['products'])}")
        if parsed["agents"]:
            console.print(f"  Agents: {', '.join(parsed['agents'])}")
        console.print()


def demo_reaction_dataset():
    """Demonstrate reaction dataset management."""
    console.print("\n[bold blue]3. Reaction Dataset Demo[/bold blue]\n")

    # Create dataset
    dataset = ReactionDataset()

    # Add some reactions
    reactions_data = [
        {
            "id": "rxn_001",
            "smiles": "CCO.CC(=O)O>>CCOC(=O)C",
            "temp": 80.0,
            "catalyst": "H2SO4",
            "yield": 85.0,
        },
        {
            "id": "rxn_002",
            "smiles": "c1ccccc1Br.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
            "temp": 100.0,
            "catalyst": "Pd(PPh3)4",
            "yield": 92.0,
        },
    ]

    rxn_parser = ReactionSMILES()

    for rxn_data in reactions_data:
        parsed = rxn_parser.parse_reaction(rxn_data["smiles"])
        conditions = ReactionConditions(
            temperature=rxn_data["temp"], catalyst=rxn_data["catalyst"]
        )

        reaction = ChemicalReaction(
            reaction_id=rxn_data["id"],
            reactants=parsed["reactants"],
            products=parsed["products"],
            agents=parsed["agents"],
            conditions=conditions,
            yield_percentage=rxn_data["yield"],
            source="manual",
        )

        dataset.add_reaction(reaction)

    console.print(f"Created dataset with {len(dataset.reactions)} reactions")

    # Get statistics
    stats = dataset.get_statistics()
    console.print("\n[bold]Dataset Statistics:[/bold]")
    console.print(f"  Number of reactions: {stats['num_reactions']}")
    temp_stats = stats["temperature_stats"]
    console.print(
        f"  Temperature range: {temp_stats['min']:.1f}°C - {temp_stats['max']:.1f}°C"
    )

    # Save to file
    output_path = Path("data/processed/demo_reactions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_dataset(output_path)
    console.print(f"\n[green]✓ Saved dataset to {output_path}[/green]")


def demo_uspto_generation():
    """Demonstrate USPTO-style data generation."""
    console.print("\n[bold blue]4. USPTO-Style Data Generation[/bold blue]\n")

    loader = USPTOLoader()

    # Generate synthetic dataset
    console.print("Generating 50 synthetic reactions...")
    dataset = loader.create_synthetic_dataset(num_reactions=50)

    console.print(f"[green]✓ Generated {len(dataset.reactions)} reactions[/green]")

    # Show sample
    sample_reaction = dataset.reactions[0]
    console.print("\n[bold]Sample Reaction:[/bold]")
    console.print(f"  ID: {sample_reaction.reaction_id}")
    console.print(f"  Reactants: {', '.join(sample_reaction.reactants)}")
    console.print(f"  Products: {', '.join(sample_reaction.products)}")
    console.print(f"  Temperature: {sample_reaction.conditions.temperature:.1f}°C")
    console.print(f"  Catalyst: {sample_reaction.conditions.catalyst or 'None'}")
    console.print(f"  Yield: {sample_reaction.yield_percentage:.1f}%")

    # Save dataset
    output_path = Path("data/processed/uspto_synthetic_50.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_dataset(output_path)
    console.print(f"\n[green]✓ Saved to {output_path}[/green]")


def demo_feature_extraction():
    """Demonstrate feature extraction for ML."""
    console.print("\n[bold blue]5. Feature Extraction for ML[/bold blue]\n")

    # Load synthetic dataset
    loader = USPTOLoader()
    dataset = loader.create_synthetic_dataset(num_reactions=20)

    # Extract features
    console.print("Extracting features from reactions...")
    builder = ReactionFeatureBuilder()
    features_df, targets = builder.build_features(dataset.reactions)

    console.print(f"\n[green]✓ Extracted features: {features_df.shape}[/green]")
    console.print(f"  Samples: {len(features_df)}")
    console.print(f"  Features: {len(features_df.columns)}")

    # Show sample features
    console.print("\n[bold]Sample Features (first 5):[/bold]")
    console.print(features_df.head().to_string())

    console.print("\n[bold]Feature Names:[/bold]")
    for i, col in enumerate(features_df.columns[:10], 1):
        console.print(f"  {i}. {col}")
    console.print(f"  ... and {len(features_df.columns) - 10} more")


def main():
    """Run all demonstrations."""
    console.print(
        "[bold magenta]Phase 2 Demo: Real Chemistry & SMILES[/bold magenta]"
    )
    console.print(
        "[magenta]Demonstrating molecular representation and USPTO integration[/magenta]\n"
    )

    demo_smiles_parsing()
    demo_reaction_parsing()
    demo_reaction_dataset()
    demo_uspto_generation()
    demo_feature_extraction()

    console.print("\n[bold green]✓ Phase 2 demonstration complete![/bold green]")
    console.print(
        "\n[yellow]Next steps:[/yellow] These features enable training ML models on real chemistry data!"
    )


if __name__ == "__main__":
    main()
