"""Tests for reaction dataset classes."""

import pytest
import tempfile
from pathlib import Path
from src.data.reaction_dataset import (
    ReactionConditions,
    ChemicalReaction,
    ReactionDataset,
)


class TestReactionConditions:
    """Test suite for ReactionConditions."""

    def test_creation(self):
        """Test conditions creation."""
        conditions = ReactionConditions(temperature=80.0, catalyst="Pd")
        assert conditions.temperature == 80.0
        assert conditions.catalyst == "Pd"
        assert conditions.solvent is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conditions = ReactionConditions(temperature=80.0)
        d = conditions.to_dict()
        assert "temperature" in d
        assert "solvent" not in d  # None values excluded

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"temperature": 80.0, "catalyst": "Pd"}
        conditions = ReactionConditions.from_dict(data)
        assert conditions.temperature == 80.0
        assert conditions.catalyst == "Pd"


class TestChemicalReaction:
    """Test suite for ChemicalReaction."""

    def test_creation(self):
        """Test reaction creation."""
        conditions = ReactionConditions(temperature=80.0)
        reaction = ChemicalReaction(
            reaction_id="test_1",
            reactants=["CCO", "CC(=O)O"],
            products=["CCOC(=O)C"],
            conditions=conditions,
        )
        assert reaction.reaction_id == "test_1"
        assert len(reaction.reactants) == 2
        assert len(reaction.products) == 1

    def test_to_reaction_smiles(self):
        """Test conversion to reaction SMILES."""
        conditions = ReactionConditions()
        reaction = ChemicalReaction(
            reaction_id="test_1",
            reactants=["CCO", "CC(=O)O"],
            products=["CCOC(=O)C"],
            conditions=conditions,
        )
        smiles = reaction.to_reaction_smiles()
        assert ">>" in smiles
        assert "CCO" in smiles

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        conditions = ReactionConditions(temperature=80.0)
        reaction = ChemicalReaction(
            reaction_id="test_1",
            reactants=["CCO"],
            products=["CC=C"],
            conditions=conditions,
            reaction_rate=0.5,
        )
        data = reaction.to_dict()
        restored = ChemicalReaction.from_dict(data)

        assert restored.reaction_id == reaction.reaction_id
        assert restored.reactants == reaction.reactants
        assert restored.reaction_rate == reaction.reaction_rate


class TestReactionDataset:
    """Test suite for ReactionDataset."""

    def test_add_reaction(self):
        """Test adding reactions to dataset."""
        dataset = ReactionDataset()
        conditions = ReactionConditions(temperature=80.0)
        reaction = ChemicalReaction(
            reaction_id="test_1",
            reactants=["CCO"],
            products=["CC=C"],
            conditions=conditions,
        )
        dataset.add_reaction(reaction)
        assert len(dataset.reactions) == 1

    def test_filter_by_temperature(self):
        """Test filtering by temperature."""
        dataset = ReactionDataset()

        # Add reactions at different temperatures
        for i, temp in enumerate([50, 75, 100]):
            conditions = ReactionConditions(temperature=temp)
            reaction = ChemicalReaction(
                reaction_id=f"test_{i}",
                reactants=["CCO"],
                products=["CC=C"],
                conditions=conditions,
            )
            dataset.add_reaction(reaction)

        # Filter for temp >= 70
        filtered = dataset.filter_by_conditions(min_temp=70)
        assert len(filtered.reactions) == 2

    def test_save_load_json(self):
        """Test JSON serialization roundtrip."""
        dataset = ReactionDataset()
        conditions = ReactionConditions(temperature=80.0)
        reaction = ChemicalReaction(
            reaction_id="test_1",
            reactants=["CCO"],
            products=["CC=C"],
            conditions=conditions,
            reaction_rate=0.5,
        )
        dataset.add_reaction(reaction)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            dataset.save_dataset(filepath)

            loaded = ReactionDataset(filepath)
            assert len(loaded.reactions) == 1
            assert loaded.reactions[0].reaction_id == "test_1"
            assert loaded.reactions[0].reaction_rate == 0.5

    def test_statistics(self):
        """Test dataset statistics calculation."""
        dataset = ReactionDataset()

        for i in range(10):
            conditions = ReactionConditions(temperature=float(50 + i * 5))
            reaction = ChemicalReaction(
                reaction_id=f"test_{i}",
                reactants=["CCO"],
                products=["CC=C"],
                conditions=conditions,
                reaction_rate=0.1 * i,
            )
            dataset.add_reaction(reaction)

        stats = dataset.get_statistics()
        assert stats["num_reactions"] == 10
        assert stats["temperature_stats"]["min"] == 50.0
        assert stats["temperature_stats"]["max"] == 95.0
