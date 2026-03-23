"""Integration-style tests for the Typer app (CLI entry point).

Uses typer.testing.CliRunner so no subprocess is spawned.
All tests run without network access or heavy ML deps.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner(mix_stderr=False)


class TestCLIEntryPoint:
    """Smoke tests for the top-level `chem-rate` command."""

    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_version_flag(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "chem-rate" in result.output

    def test_no_args_shows_banner_or_help(self):
        """With no sub-command the app should not crash."""
        result = runner.invoke(app, [])
        # Exit 0 (banner) or 2 (typer no-args-is-help) — both OK
        assert result.exit_code in (0, 2)


class TestInfoCommands:
    """Tests for `chem-rate info` sub-commands."""

    def test_info_system_exits_zero(self):
        result = runner.invoke(app, ["info", "system"])
        assert result.exit_code == 0
        assert "Python" in result.output

    def test_info_deps_exits_zero(self):
        result = runner.invoke(app, ["info", "deps"])
        assert result.exit_code == 0
        # Should list at least typer and rich
        assert "typer" in result.output
        assert "rich" in result.output

    def test_info_help(self):
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0


class TestValidateCommands:
    """Tests for `chem-rate validate` sub-commands."""

    def test_validate_smiles_valid(self):
        result = runner.invoke(app, ["validate", "smiles", "CCO"])
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_validate_smiles_reaction(self):
        result = runner.invoke(app, ["validate", "smiles", "CCO>>CC=O"])
        assert result.exit_code == 0

    def test_validate_smiles_empty_exits_nonzero(self):
        result = runner.invoke(app, ["validate", "smiles", ""])
        assert result.exit_code != 0

    def test_validate_smiles_bracket_mismatch_exits_nonzero(self):
        result = runner.invoke(app, ["validate", "smiles", "[NH4"])
        assert result.exit_code != 0

    def test_validate_smiles_paren_mismatch_exits_nonzero(self):
        result = runner.invoke(app, ["validate", "smiles", "CC(=O"])
        assert result.exit_code != 0

    def test_validate_temperature_valid(self):
        result = runner.invoke(app, ["validate", "temperature", "500"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_temperature_negative_exits_nonzero(self):
        result = runner.invoke(app, ["validate", "temperature", "-10"])
        assert result.exit_code != 0

    def test_validate_temperature_above_max_exits_nonzero(self):
        result = runner.invoke(app, ["validate", "temperature", "9999"])
        assert result.exit_code != 0

    def test_validate_temperature_custom_range(self):
        result = runner.invoke(
            app, ["validate", "temperature", "150", "--min", "100", "--max", "200"]
        )
        assert result.exit_code == 0

    def test_validate_help(self):
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
