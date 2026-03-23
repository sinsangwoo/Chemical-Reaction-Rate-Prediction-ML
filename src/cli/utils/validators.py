"""Input validators for CLI commands.

Design goals
------------
* Never crash if optional heavy deps (RDKit, PyTorch) are absent.
* Cover all realistic edge cases a researcher might type.
* Raise typed exceptions so commands can catch and format them nicely.

Edge cases handled
------------------
SMILES
  - Empty / whitespace-only string
  - Non-ASCII / binary garbage
  - Extremely long strings (> 2000 chars → likely a paste error)
  - Missing reaction arrow in reaction-SMILES (>>)
  - Invalid atom symbols caught by heuristic (no RDKit) or RDKit itself

Temperature
  - Exactly 0 K (absolute zero — physically valid but warn)
  - Negative Kelvin
  - NaN / Inf
  - Above 5000 K (above typical chemical simulation range → warn)
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

MAX_SMILES_LEN = 2000
_SMILES_VALID_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\/\\.%:\*>]+$")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CLIValidationError(ValueError):
    """Base class for CLI validation errors."""


class SMILESValidationError(CLIValidationError):
    """Raised when a SMILES string fails validation."""


class TemperatureValidationError(CLIValidationError):
    """Raised when a temperature value is out of range or unphysical."""


# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------


def validate_smiles(
    smiles: str,
    *,
    verbose: bool = False,
    allow_reaction: bool = True,
) -> dict[str, Any]:
    """Validate a SMILES (or reaction SMILES) string.

    Parameters
    ----------
    smiles:
        The SMILES string to validate.
    verbose:
        If True, include atom-level detail in the returned dict.
    allow_reaction:
        If True, reaction SMILES with '>>' separators are accepted.

    Returns
    -------
    dict with keys: atoms, bonds, engine, atom_list (if verbose).

    Raises
    ------
    SMILESValidationError on any validation failure.
    """
    # --- Basic sanity -------------------------------------------------------
    if not smiles or not smiles.strip():
        raise SMILESValidationError("SMILES string must not be empty.")

    smiles = smiles.strip()

    # Non-ASCII check (catches binary paste / encoding errors)
    for ch in smiles:
        cat = unicodedata.category(ch)
        if ord(ch) > 127 and cat not in ("Ll", "Lu", "Nd"):
            raise SMILESValidationError(
                f"Non-ASCII character {ch!r} detected. "
                "SMILES must contain only printable ASCII."
            )

    if len(smiles) > MAX_SMILES_LEN:
        raise SMILESValidationError(
            f"SMILES is unusually long ({len(smiles)} chars > {MAX_SMILES_LEN}). "
            "This may be a paste error."
        )

    # Reaction SMILES handling
    if ">>" in smiles:
        if not allow_reaction:
            raise SMILESValidationError(
                "Reaction SMILES ('>>') are not allowed here. "
                "Provide a single molecule SMILES."
            )
        parts = smiles.split(">>")
        if len(parts) != 2:  # noqa: PLR2004
            raise SMILESValidationError(
                "Reaction SMILES must have exactly one '>>' separator "
                f"(found {len(parts) - 1})."
            )
        # Validate each part independently
        for part in parts:
            if part.strip():
                _validate_single_smiles(part.strip())
        return {"atoms": "n/a (reaction)", "bonds": "n/a", "engine": "heuristic"}

    return _validate_single_smiles(smiles, verbose=verbose)


def _validate_single_smiles(
    smiles: str, *, verbose: bool = False
) -> dict[str, Any]:
    """Validate a single-molecule SMILES string."""
    # Try RDKit first (accurate)
    try:
        from rdkit import Chem  # type: ignore[import]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SMILESValidationError(
                f"RDKit could not parse SMILES: {smiles!r}"
            )
        atom_list = [a.GetSymbol() for a in mol.GetAtoms()] if verbose else None
        return {
            "atoms": mol.GetNumAtoms(),
            "bonds": mol.GetNumBonds(),
            "engine": "rdkit",
            "atom_list": atom_list,
        }
    except ImportError:
        pass  # Fall through to heuristic

    # Heuristic fallback (no RDKit)
    if not _SMILES_VALID_CHARS.match(smiles):
        bad = [
            ch for ch in smiles if not re.match(r"[A-Za-z0-9@+\-\[\]\(\)=#\/\\.%:\*>]", ch)
        ]
        raise SMILESValidationError(
            f"Invalid character(s) in SMILES: {bad!r}. "
            "Install RDKit for full validation: conda install -c conda-forge rdkit"
        )

    # Count bracket atoms and open/close parity
    open_brackets = smiles.count("[")
    close_brackets = smiles.count("]")
    if open_brackets != close_brackets:
        raise SMILESValidationError(
            f"Mismatched brackets in SMILES: {open_brackets} '[' vs {close_brackets} ']'."
        )

    open_parens = smiles.count("(")
    close_parens = smiles.count(")")
    if open_parens != close_parens:
        raise SMILESValidationError(
            f"Mismatched parentheses in SMILES: {open_parens} '(' vs {close_parens} ')'. "
        )

    return {
        "atoms": "n/a",
        "bonds": "n/a",
        "engine": "heuristic (install RDKit for exact counts)",
    }


# ---------------------------------------------------------------------------
# Temperature validation
# ---------------------------------------------------------------------------

ABSOLUTE_ZERO_K = 0.0
_WARN_HIGH_K = 5000.0


def validate_temperature(
    temp_k: float,
    *,
    min_k: float = ABSOLUTE_ZERO_K,
    max_k: float = _WARN_HIGH_K,
) -> None:
    """Validate a temperature value in Kelvin.

    Parameters
    ----------
    temp_k: Temperature in Kelvin.
    min_k:  Minimum acceptable value (default: 0 K — absolute zero).
    max_k:  Maximum acceptable value (default: 5000 K).

    Raises
    ------
    TemperatureValidationError for NaN, Inf, or out-of-range values.
    """
    if math.isnan(temp_k):
        raise TemperatureValidationError("Temperature must be a real number, got NaN.")

    if math.isinf(temp_k):
        raise TemperatureValidationError(
            f"Temperature must be finite, got {'∞' if temp_k > 0 else '-∞'}."
        )

    if temp_k < min_k:
        if temp_k < 0:
            raise TemperatureValidationError(
                f"Temperature {temp_k} K is below absolute zero (0 K). "
                "Negative Kelvin values are unphysical."
            )
        raise TemperatureValidationError(
            f"Temperature {temp_k} K is below the allowed minimum of {min_k} K."
        )

    if temp_k > max_k:
        raise TemperatureValidationError(
            f"Temperature {temp_k} K exceeds the allowed maximum of {max_k} K. "
            "This is beyond the typical chemical simulation range. "
            "Pass --max to override."
        )
