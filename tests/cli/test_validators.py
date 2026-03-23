"""Unit tests for src/cli/utils/validators.py.

Covers every edge case documented in the validators module.
All tests run without RDKit or PyTorch (heuristic path only).
"""

from __future__ import annotations

import math

import pytest

from src.cli.utils.validators import (
    SMILESValidationError,
    TemperatureValidationError,
    validate_smiles,
    validate_temperature,
)


# ===========================================================================
# SMILES validation
# ===========================================================================


class TestValidateSmiles:
    """Tests for validate_smiles()."""

    # --- Happy path --------------------------------------------------------

    def test_simple_ethanol(self):
        result = validate_smiles("CCO")
        assert result["engine"] in ("rdkit", "heuristic (install RDKit for exact counts)")

    def test_reaction_smiles_accepted_by_default(self):
        result = validate_smiles("CCO>>CC=O")
        assert "reaction" in result["atoms"]

    def test_complex_molecule(self):
        # Aspirin SMILES
        result = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None

    def test_charged_atom_brackets(self):
        result = validate_smiles("[NH4+]")
        assert result is not None

    def test_isotope_label(self):
        result = validate_smiles("[13C]")
        assert result is not None

    # --- Edge cases: empty / whitespace ------------------------------------

    def test_empty_string_raises(self):
        with pytest.raises(SMILESValidationError, match="must not be empty"):
            validate_smiles("")

    def test_whitespace_only_raises(self):
        with pytest.raises(SMILESValidationError, match="must not be empty"):
            validate_smiles("   ")

    # --- Edge cases: length -----------------------------------------------

    def test_too_long_smiles_raises(self):
        long_smiles = "C" * 2001
        with pytest.raises(SMILESValidationError, match="unusually long"):
            validate_smiles(long_smiles)

    def test_max_length_exactly_allowed(self):
        # Exactly at limit should not raise
        smiles = "C" * 2000
        result = validate_smiles(smiles)
        assert result is not None

    # --- Edge cases: invalid characters ------------------------------------

    def test_non_ascii_garbage_raises(self):
        # Null byte — definitely invalid
        with pytest.raises(SMILESValidationError):
            validate_smiles("CCO\x00")

    def test_bracket_mismatch_raises(self):
        with pytest.raises(SMILESValidationError, match="bracket"):
            validate_smiles("[NH4")

    def test_paren_mismatch_raises(self):
        with pytest.raises(SMILESValidationError, match="parenthes"):
            validate_smiles("CC(=O")

    # --- Edge cases: reaction SMILES ---------------------------------------

    def test_reaction_smiles_disallowed_when_flag_false(self):
        with pytest.raises(SMILESValidationError, match="not allowed"):
            validate_smiles("CCO>>CC=O", allow_reaction=False)

    def test_double_arrow_reaction_smiles(self):
        # Two '>>' — three parts — should fail
        with pytest.raises(SMILESValidationError, match="exactly one"):
            validate_smiles("A>>B>>C")

    def test_empty_reagent_side_still_valid(self):
        # Reactant with no product (unusual but syntactically OK)
        result = validate_smiles("CCO>>")
        assert result is not None


# ===========================================================================
# Temperature validation
# ===========================================================================


class TestValidateTemperature:
    """Tests for validate_temperature()."""

    # --- Happy path --------------------------------------------------------

    def test_room_temperature(self):
        validate_temperature(298.15)  # should not raise

    def test_absolute_zero_allowed(self):
        validate_temperature(0.0)  # edge but valid

    def test_exact_maximum_allowed(self):
        validate_temperature(5000.0)  # right at the boundary

    def test_custom_range(self):
        validate_temperature(1.0, min_k=1.0, max_k=2.0)

    # --- Edge cases: unphysical values ------------------------------------

    def test_negative_kelvin_raises(self):
        with pytest.raises(TemperatureValidationError, match="absolute zero"):
            validate_temperature(-1.0)

    def test_nan_raises(self):
        with pytest.raises(TemperatureValidationError, match="NaN"):
            validate_temperature(math.nan)

    def test_positive_inf_raises(self):
        with pytest.raises(TemperatureValidationError, match="finite"):
            validate_temperature(math.inf)

    def test_negative_inf_raises(self):
        with pytest.raises(TemperatureValidationError, match="finite"):
            validate_temperature(-math.inf)

    def test_above_max_raises(self):
        with pytest.raises(TemperatureValidationError, match="exceeds"):
            validate_temperature(5001.0)

    def test_below_custom_min_raises(self):
        with pytest.raises(TemperatureValidationError, match="below the allowed minimum"):
            validate_temperature(99.0, min_k=100.0)

    def test_above_custom_max_raises(self):
        with pytest.raises(TemperatureValidationError, match="exceeds"):
            validate_temperature(201.0, min_k=0.0, max_k=200.0)
