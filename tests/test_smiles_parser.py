"""Tests for SMILES parser."""

import pytest
from src.data.smiles_parser import SMILESParser, ReactionSMILES


class TestSMILESParser:
    """Test suite for SMILESParser."""

    def test_valid_simple_smiles(self):
        """Test validation of simple SMILES."""
        parser = SMILESParser()
        assert parser.is_valid_smiles("CCO") is True  # Ethanol
        assert parser.is_valid_smiles("CC(=O)O") is True  # Acetic acid
        assert parser.is_valid_smiles("c1ccccc1") is True  # Benzene

    def test_invalid_smiles(self):
        """Test rejection of invalid SMILES."""
        parser = SMILESParser()
        assert parser.is_valid_smiles("") is False
        assert parser.is_valid_smiles("((C") is False  # Unbalanced parentheses
        assert parser.is_valid_smiles(None) is False

    def test_extract_atoms(self):
        """Test atom extraction from SMILES."""
        parser = SMILESParser()
        atoms = parser.extract_atoms("CCO")
        assert "C" in atoms
        assert "O" in atoms
        assert len([a for a in atoms if a == "C"]) == 2

    def test_count_atoms(self):
        """Test atom counting."""
        parser = SMILESParser()
        counts = parser.count_atoms("CCO")
        assert counts["C"] == 2
        assert counts["O"] == 1

    def test_two_letter_atoms(self):
        """Test recognition of two-letter atoms (Cl, Br)."""
        parser = SMILESParser()
        atoms = parser.extract_atoms("CCCl")
        assert "Cl" in atoms
        assert atoms.count("C") == 2

    def test_molecular_weight_estimate(self):
        """Test molecular weight estimation."""
        parser = SMILESParser()
        # Benzene (C6H6) should be around 78 g/mol
        mw = parser.get_molecular_weight_estimate("c1ccccc1")
        assert 70 < mw < 90

    def test_extract_features(self):
        """Test comprehensive feature extraction."""
        parser = SMILESParser()
        features = parser.extract_features("c1ccccc1")
        assert "smiles" in features
        assert "atom_counts" in features
        assert "has_aromatic" in features
        assert features["has_aromatic"] is True


class TestReactionSMILES:
    """Test suite for ReactionSMILES."""

    def test_parse_simple_reaction(self):
        """Test parsing of simple reaction SMILES."""
        parser = ReactionSMILES()
        result = parser.parse_reaction("CCO.CC(=O)O>>CCOC(=O)C")

        assert len(result["reactants"]) == 2
        assert len(result["products"]) == 1
        assert "CCO" in result["reactants"]
        assert "CC(=O)O" in result["reactants"]

    def test_parse_with_agents(self):
        """Test parsing reaction with agents/catalysts."""
        parser = ReactionSMILES()
        result = parser.parse_reaction("CCO>H2SO4>CC=C")

        assert len(result["reactants"]) == 1
        assert len(result["agents"]) == 1
        assert len(result["products"]) == 1
        assert "H2SO4" in result["agents"]

    def test_validate_reaction(self):
        """Test reaction validation."""
        parser = ReactionSMILES()
        assert parser.validate_reaction("CCO.CC(=O)O>>CCOC(=O)C") is True
        assert parser.validate_reaction("invalid") is False
        assert parser.validate_reaction("") is False

    def test_extract_reaction_features(self):
        """Test reaction feature extraction."""
        parser = ReactionSMILES()
        features = parser.extract_reaction_features("CCO.CC(=O)O>>CCOC(=O)C")

        assert features["num_reactants"] == 2
        assert features["num_products"] == 1
        assert "reactant_features" in features
        assert "product_features" in features
