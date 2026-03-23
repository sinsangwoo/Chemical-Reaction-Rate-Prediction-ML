"""Chemical-CLI: Terminal-based interactive workflow for chem-rate.

Entry point: `chem-rate` (registered via pyproject.toml console_scripts).
Phase 1 delivers the scaffold + helper utilities.
Phase 2 will add `simulate`, `predict`, and `benchmark` commands.
"""

__version__ = "0.1.0"
__all__ = ["app"]

from src.cli.main import app
