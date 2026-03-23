"""Tests for ASCII art helpers."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from src.cli.utils.ascii_art import plot_energy_barrier_stub, print_banner


def _capture(fn, *args, **kwargs) -> str:
    """Run a function that calls console.print and capture its output."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, highlight=False)
    # Monkey-patch temporarily
    import src.cli.utils.ascii_art as mod

    original = mod.console
    mod.console = console
    try:
        fn(*args, **kwargs)
    finally:
        mod.console = original
    return buf.getvalue()


class TestPrintBanner:
    def test_banner_contains_chem(self):
        output = _capture(print_banner)
        # Banner or subtitle should mention the product
        assert "Chemical" in output or "chem" in output.lower()

    def test_banner_no_exception(self):
        print_banner()  # Should not raise


class TestEnergyBarrierStub:
    def test_stub_no_exception(self):
        plot_energy_barrier_stub("CCO>>CC=O", 500.0)

    def test_stub_output_contains_temp(self):
        output = _capture(plot_energy_barrier_stub, "CCO>>CC=O", 500.0)
        assert "500" in output

    def test_stub_output_contains_smiles(self):
        output = _capture(plot_energy_barrier_stub, "CCO>>CC=O", 500.0)
        assert "CCO" in output
