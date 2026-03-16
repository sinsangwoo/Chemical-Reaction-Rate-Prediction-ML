"""Tests for the ReactionNetworkDataset."""

import math
import pytest

from src.data.reaction_network_dataset import (
    ReactionNetworkDataset,
    ReactionNetworkSample,
)


class TestReactionNetworkSample:
    def test_log_a_values(self):
        s = ReactionNetworkSample(
            reactant_smiles=["A"],
            product_smiles=["B"],
            intermediate_smiles=[],
            ea_values=[80.0],
            a_values=[1e13],
            temperature=500.0,
        )
        assert abs(s.log_a_values[0] - math.log(1e13)) < 1e-6


class TestReactionNetworkDataset:
    def test_synthetic_length(self):
        ds = ReactionNetworkDataset.synthetic(n_samples=50)
        assert len(ds) == 50

    def test_getitem(self):
        ds = ReactionNetworkDataset.synthetic(n_samples=10)
        sample = ds[0]
        assert isinstance(sample, ReactionNetworkSample)
        assert len(sample.ea_values) >= 1

    def test_split_sizes(self):
        ds = ReactionNetworkDataset.synthetic(n_samples=100)
        train, val = ds.split(train_frac=0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_summary_string(self):
        ds = ReactionNetworkDataset.synthetic(n_samples=20)
        s = ds.summary()
        assert "20" in s

    def test_empty_dataset(self):
        ds = ReactionNetworkDataset()
        assert len(ds) == 0
        assert "empty" in ds.summary()
