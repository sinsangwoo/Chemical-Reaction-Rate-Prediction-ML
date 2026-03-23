"""Main Typer application — `chem-rate` CLI entry point.

Phase 1 registers the app and stub commands so the package is
immediately installable and explorable. Full simulation / prediction
commands arrive in Phase 2.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.cli.commands import info as info_cmd
from src.cli.commands import validate as validate_cmd
from src.cli.utils.ascii_art import print_banner
from src.cli import __version__

console = Console()

app = typer.Typer(
    name="chem-rate",
    help="""
    Chemical Reaction Rate Prediction CLI.

    Predict reaction rate constants, visualise energy barriers,
    and benchmark ML models — all from your terminal.
    """,
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# --- Sub-command groups -------------------------------------------------
app.add_typer(
    info_cmd.app,
    name="info",
    help="Show system / package information.",
)
app.add_typer(
    validate_cmd.app,
    name="validate",
    help="Validate inputs (SMILES strings, temperature ranges, …).",
)


# --- Top-level commands -------------------------------------------------

@app.command()
def version() -> None:
    """Print the chem-rate version and exit."""
    console.print(
        Panel(
            Text(f"chem-rate  v{__version__}", justify="center", style="bold cyan"),
            subtitle="Chemical Reaction Rate Prediction CLI",
            border_style="cyan",
        )
    )


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    show_version: Optional[bool] = typer.Option(
        None, "--version", "-V", help="Print version and exit.", is_eager=True
    ),
) -> None:
    """Global callback — shows the banner when called with no sub-command."""
    if show_version:
        version()
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(
            "Run [bold cyan]chem-rate --help[/bold cyan] to see available commands.\n"
        )


def entry_point() -> None:  # pragma: no cover
    """Setuptools console_scripts entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    entry_point()
