"""SMILES notation parser and molecular feature extractor."""

from typing import Dict, List, Optional, Tuple
import re


class SMILESParser:
    """Parse and validate SMILES (Simplified Molecular Input Line Entry System) strings."""

    # Basic SMILES syntax patterns
    ATOM_PATTERN = r"[BCNOPSFIbcnops]|Cl|Br"
    BOND_PATTERN = r"[-=#$:\\/]"
    BRANCH_PATTERN = r"[\(\)]"
    RING_PATTERN = r"[0-9%]"

    def __init__(self):
        """Initialize SMILES parser."""
        self.valid_atoms = {
            "C",
            "N",
            "O",
            "S",
            "P",
            "F",
            "Cl",
            "Br",
            "I",
            "B",
            "c",
            "n",
            "o",
            "s",
            "p",
            "b",
        }

    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string has valid syntax.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not smiles or not isinstance(smiles, str):
            return False

        # Check for basic validity
        if len(smiles) > 500:  # Reasonable length limit
            return False

        # Check balanced parentheses
        if smiles.count("(") != smiles.count(")"):
            return False

        return True

    def extract_atoms(self, smiles: str) -> List[str]:
        """Extract all atoms from a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            List of atom symbols
        """
        atoms = []
        i = 0
        while i < len(smiles):
            # Check for two-letter atoms (Cl, Br)
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.valid_atoms:
                atoms.append(smiles[i : i + 2])
                i += 2
            # Check for single-letter atoms
            elif smiles[i] in self.valid_atoms:
                atoms.append(smiles[i])
                i += 1
            else:
                i += 1
        return atoms

    def count_atoms(self, smiles: str) -> Dict[str, int]:
        """Count occurrences of each atom type.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary mapping atom symbols to counts
        """
        atoms = self.extract_atoms(smiles)
        counts = {}
        for atom in atoms:
            atom_upper = atom.upper()
            counts[atom_upper] = counts.get(atom_upper, 0) + 1
        return counts

    def get_molecular_weight_estimate(self, smiles: str) -> float:
        """Estimate molecular weight from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Estimated molecular weight in g/mol
        """
        # Approximate atomic weights
        weights = {
            "C": 12.01,
            "H": 1.008,
            "N": 14.01,
            "O": 16.00,
            "S": 32.07,
            "P": 30.97,
            "F": 19.00,
            "CL": 35.45,
            "BR": 79.90,
            "I": 126.90,
            "B": 10.81,
        }

        atom_counts = self.count_atoms(smiles)
        total_weight = 0.0

        for atom, count in atom_counts.items():
            total_weight += weights.get(atom, 0) * count

        # Estimate hydrogen count (very rough approximation)
        # For organic molecules, H count ≈ 2×C + 2
        h_estimate = atom_counts.get("C", 0) * 2 + 2
        total_weight += h_estimate * weights["H"]

        return total_weight

    def extract_features(self, smiles: str) -> Dict[str, any]:
        """Extract comprehensive features from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular features
        """
        features = {
            "smiles": smiles,
            "length": len(smiles),
            "atom_counts": self.count_atoms(smiles),
            "estimated_mw": self.get_molecular_weight_estimate(smiles),
            "num_rings": self._count_rings(smiles),
            "num_branches": smiles.count("("),
            "has_aromatic": any(c in smiles for c in "cnops"),
        }
        return features

    def _count_rings(self, smiles: str) -> int:
        """Count number of rings in SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Approximate number of rings
        """
        # Count ring closure digits (rough approximation)
        ring_digits = sum(1 for c in smiles if c.isdigit())
        return ring_digits // 2  # Each ring uses 2 digits


class ReactionSMILES:
    """Handle chemical reaction SMILES notation."""

    def __init__(self):
        """Initialize reaction SMILES handler."""
        self.parser = SMILESParser()

    def parse_reaction(self, reaction_smiles: str) -> Dict[str, List[str]]:
        """Parse reaction SMILES into reactants, agents, and products.

        Args:
            reaction_smiles: Reaction SMILES string (format: reactants>agents>products)

        Returns:
            Dictionary with 'reactants', 'agents', and 'products' lists
        """
        if ">>" in reaction_smiles:
            # Standard format: reactants>>products
            parts = reaction_smiles.split(">>")
            reactants = parts[0].split(".") if parts[0] else []
            products = parts[1].split(".") if len(parts) > 1 else []
            agents = []
        elif ">" in reaction_smiles:
            # Extended format: reactants>agents>products
            parts = reaction_smiles.split(">")
            reactants = parts[0].split(".") if parts[0] else []
            agents = parts[1].split(".") if len(parts) > 1 and parts[1] else []
            products = parts[2].split(".") if len(parts) > 2 else []
        else:
            raise ValueError(f"Invalid reaction SMILES format: {reaction_smiles}")

        return {
            "reactants": [r.strip() for r in reactants if r.strip()],
            "agents": [a.strip() for a in agents if a.strip()],
            "products": [p.strip() for p in products if p.strip()],
        }

    def validate_reaction(self, reaction_smiles: str) -> bool:
        """Validate a reaction SMILES string.

        Args:
            reaction_smiles: Reaction SMILES string

        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = self.parse_reaction(reaction_smiles)

            # Check that we have reactants and products
            if not parsed["reactants"] or not parsed["products"]:
                return False

            # Validate each SMILES component
            all_smiles = (
                parsed["reactants"] + parsed["agents"] + parsed["products"]
            )
            return all(self.parser.is_valid_smiles(s) for s in all_smiles)

        except Exception:
            return False

    def extract_reaction_features(self, reaction_smiles: str) -> Dict[str, any]:
        """Extract features from a reaction SMILES.

        Args:
            reaction_smiles: Reaction SMILES string

        Returns:
            Dictionary of reaction features
        """
        parsed = self.parse_reaction(reaction_smiles)

        features = {
            "reaction_smiles": reaction_smiles,
            "num_reactants": len(parsed["reactants"]),
            "num_products": len(parsed["products"]),
            "num_agents": len(parsed["agents"]),
            "reactant_features": [
                self.parser.extract_features(s) for s in parsed["reactants"]
            ],
            "product_features": [
                self.parser.extract_features(s) for s in parsed["products"]
            ],
        }

        return features
