"""chem-rate validate — input validation commands.

Provides lightweight, RDKit-optional SMILES and temperature
validation so users get fast feedback before running heavy ML.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from src.cli.utils.validators import (
    SMILESValidationError,
    TemperatureValidationError,
    validate_smiles,
    validate_temperature,
)

console = Console()
app = typer.Typer(
    help="Validate inputs before running simulations.", rich_markup_mode="rich"
)


@app.command("smiles")
def validate_smiles_cmd(
    smiles: str = typer.Argument(..., help="SMILES string to validate, e.g. 'CCO'"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show parsed atom info."),
) -> None:
    """Validate a SMILES string (RDKit if available, else heuristic).

    Examples:

        chem-rate validate smiles 'CCO'

        chem-rate validate smiles 'CCO>>CC=O' --verbose
    """
    try:
        result = validate_smiles(smiles, verbose=verbose)
    except SMILESValidationError as exc:
        console.print(f"[bold red]Invalid SMILES:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[green]✔  Valid SMILES[/green]\n\n"
            f"Input : [cyan]{smiles}[/cyan]\n"
            f"Atoms : {result.get('atoms', 'n/a')}\n"
            f"Bonds : {result.get('bonds', 'n/a')}\n"
            f"Engine: {result.get('engine', 'heuristic')}",
            title="SMILES Validation",
            border_style="green",
        )
    )
    if verbose and result.get("atom_list"):
        console.print("[bold]Atom list:[/bold]", result["atom_list"])


@app.command("temperature")
def validate_temperature_cmd(
    temp: float = typer.Argument(..., help="Temperature in Kelvin."),
    min_k: float = typer.Option(0.0, "--min", help="Minimum allowed temperature (K)."),
    max_k: float = typer.Option(5000.0, "--max", help="Maximum allowed temperature (K)."),
) -> None:
    """Validate a temperature value (Kelvin).

    Examples:

        chem-rate validate temperature 500

        chem-rate validate temperature -10   # will fail
    """
    try:
        validate_temperature(temp, min_k=min_k, max_k=max_k)
    except TemperatureValidationError as exc:
        console.print(f"[bold red]Invalid temperature:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]✔  Temperature [cyan]{temp} K[/cyan] is valid "
        f"(range {min_k}–{max_k} K).[/green]"
    )
