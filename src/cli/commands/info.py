"""chem-rate info — system & package information commands."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Show system / package information.", rich_markup_mode="rich")


@app.command("system")
def system_info() -> None:
    """Display Python, OS, and key dependency versions."""
    table = Table(title="System Information", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="bold", min_width=22)
    table.add_column("Version / Value", style="green")

    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Architecture", platform.machine())

    # Optional heavy deps — import failures are non-fatal
    for pkg, import_name in [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("rdkit", "rdkit"),
        ("plotext", "plotext"),
        ("rich", "rich"),
        ("typer", "typer"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
        except ImportError:
            ver = "[red]not installed[/red]"
        table.add_row(pkg, ver)

    console.print(table)


@app.command("deps")
def check_deps() -> None:
    """Check which optional heavy dependencies are available."""
    required = ["typer", "rich", "plotext", "numpy", "pandas"]
    optional = ["torch", "torch_geometric", "rdkit", "sklearn"]

    console.print("\n[bold]Required dependencies[/bold]")
    _print_dep_status(required)

    console.print("\n[bold]Optional dependencies[/bold]")
    _print_dep_status(optional)
    console.print()


def _print_dep_status(packages: list[str]) -> None:
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            console.print(f"  [green]✔[/green]  {pkg:<20} {ver}")
        except ImportError:
            console.print(f"  [red]✘[/red]  {pkg:<20} [red]missing[/red]")
