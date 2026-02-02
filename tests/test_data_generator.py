"""Tests for data generation module."""

import pytest
import numpy as np
import pandas as pd
from src.data.data_generator import ArrheniusDataGenerator


class TestArrheniusDataGenerator:
    """Test suite for ArrheniusDataGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = ArrheniusDataGenerator()
        assert generator.A == 1e5
        assert generator.Ea == 40000
        assert generator.R == 8.314

    def test_generate_data_shape(self):
        """Test generated data has correct shape."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data(num_samples=100)
        assert len(data) == 100
        assert len(data.columns) == 4

    def test_generate_data_columns(self):
        """Test generated data has correct columns."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data()
        expected_cols = [
            "temperature_C",
            "concentration_mol_L",
            "catalyst",
            "reaction_rate_mol_L_s",
        ]
        assert list(data.columns) == expected_cols

    def test_temperature_range(self):
        """Test temperature values are within specified range."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data(temp_range=(20, 80))
        assert data["temperature_C"].min() >= 20
        assert data["temperature_C"].max() <= 80

    def test_concentration_range(self):
        """Test concentration values are within specified range."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data(conc_range=(0.5, 2.0))
        assert data["concentration_mol_L"].min() >= 0.5
        assert data["concentration_mol_L"].max() <= 2.0

    def test_catalyst_binary(self):
        """Test catalyst values are binary (0 or 1)."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data()
        assert set(data["catalyst"].unique()).issubset({0, 1})

    def test_reaction_rate_positive(self):
        """Test reaction rates are non-negative."""
        generator = ArrheniusDataGenerator()
        data = generator.generate_data()
        assert (data["reaction_rate_mol_L_s"] >= 0).all()

    def test_reproducibility(self):
        """Test data generation is reproducible with same seed."""
        gen1 = ArrheniusDataGenerator(random_seed=42)
        gen2 = ArrheniusDataGenerator(random_seed=42)
        data1 = gen1.generate_data(num_samples=50)
        data2 = gen2.generate_data(num_samples=50)
        pd.testing.assert_frame_equal(data1, data2)
