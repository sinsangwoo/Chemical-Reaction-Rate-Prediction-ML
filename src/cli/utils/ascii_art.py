"""ASCII art helpers: banner and energy-barrier stub for Phase 1.

Phase 2 will replace `plot_energy_barrier_stub` with a full
plotext-powered Arrhenius energy profile renderer.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from src.cli import __version__

console = Console()

_BANNER = r"""
  ██████╗██╗  ██╗███████╗███╗   ███╗      ██████╗  █████╗ ████████╗███████╗
 ██╔════╝██║  ██║██╔════╝████╗ ████║      ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
 ██║     ███████║█████╗  ██╔████╔██║█████╗██████╔╝███████║   ██║   █████╗  
 ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║╚════╝██╔══██╗██╔══██║   ██║   ██╔══╝  
 ╚██████╗██║  ██║███████╗██║ ╚═╝ ██║      ██║  ██║██║  ██║   ██║   ███████╗
  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""


def print_banner() -> None:
    """Print the ASCII banner to the terminal."""
    console.print(Text(_BANNER, style="bold cyan"))
    console.print(
        f"  Chemical Reaction Rate Prediction CLI  "
        f"[dim]v{__version__}[/dim]\n",
        justify="center",
    )


def plot_energy_barrier_stub(smiles: str, temp_k: float) -> None:
    """Placeholder energy barrier ASCII chart (Phase 1 stub).

    Phase 2 will replace this with a real plotext curve derived from
    the Arrhenius / GNN prediction pipeline.

    Parameters
    ----------
    smiles: Reaction SMILES string.
    temp_k: Reaction temperature in Kelvin.
    """
    console.print(
        "\n[bold cyan]Energy Barrier Profile[/bold cyan] "
        f"[dim](stub — full chart in Phase 2)[/dim]\n"
    )
    # Build a rough ASCII mountain shape
    width = 60
    peak = 20
    lines = []
    for row in range(peak, -1, -1):
        threshold = peak - row
        left = int(width * 0.35)
        right = int(width * 0.65)
        line = [" "] * width
        # Draw the energy curve as a mountain
        for col in range(width):
            if col < left:
                height = max(0, threshold - (left - col))
            elif col < right:
                centre = (left + right) // 2
                distance = abs(col - centre)
                height = threshold - distance // 2
            else:
                height = max(0, threshold - (col - right))
            if height >= threshold:
                line[col] = "*"
        lines.append("".join(line))

    console.print("  Ea (arb.)")
    console.print("  │")
    for ln in lines:
        console.print(f"  │ {ln}")
    console.print("  └" + "─" * (width + 2))
    console.print(f"    Reactants  →  TS  →  Products      @ {temp_k} K")
    console.print(f"    Reaction: [cyan]{smiles}[/cyan]\n")
