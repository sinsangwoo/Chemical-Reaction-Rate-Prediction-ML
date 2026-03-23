"""Shared Rich console instance and helper print functions.

Import `console` from here instead of constructing a new Console()
in every module — this ensures a single stderr-aware, colour-capable
instance across the whole CLI.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "muted": "dim",
        "highlight": "bold magenta",
    }
)

console = Console(theme=_THEME)
err_console = Console(stderr=True, theme=_THEME)


def print_info(msg: str) -> None:
    """Print an informational message."""
    console.print(f"[info]ℹ  {msg}[/info]")


def print_success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[success]✔  {msg}[/success]")


def print_warning(msg: str) -> None:
    """Print a warning (to stdout so it appears inline with results)."""
    console.print(f"[warning]⚠  {msg}[/warning]")


def print_error(msg: str) -> None:
    """Print an error to stderr."""
    err_console.print(f"[error]✘  {msg}[/error]")
