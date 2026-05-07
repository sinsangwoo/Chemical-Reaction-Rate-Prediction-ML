"""Dataset classes for chemical reaction data with SMILES support."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from .smiles_parser import ReactionSMILES, SMILESParser


@dataclass
class ReactionConditions:
    """Container for reaction conditions."""

    temperature: Optional[float] = None  # Celsius
    pressure: Optional[float] = None  # atm
    solvent: Optional[str] = None
    catalyst: Optional[str] = None
    time: Optional[float] = None  # seconds
    ph: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReactionConditions":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ChemicalReaction:
    """Container for a single chemical reaction."""

    reaction_id: str
    reactants: List[str]  # SMILES strings
    products: List[str]  # SMILES strings
    conditions: ReactionConditions
    reaction_rate: Optional[float] = None  # mol/L·s
    yield_percentage: Optional[float] = None
    agents: Optional[List[str]] = None  # Catalysts, solvents
    source: Optional[str] = None  # Dataset source

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "reaction_id": self.reaction_id,
            "reactants": self.reactants,
            "products": self.products,
            "conditions": self.conditions.to_dict(),
            "reaction_rate": self.reaction_rate,
            "yield_percentage": self.yield_percentage,
            "agents": self.agents,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChemicalReaction":
        """Create from dictionary."""
        conditions = ReactionConditions.from_dict(data.get("conditions", {}))
        return cls(
            reaction_id=data["reaction_id"],
            reactants=data["reactants"],
            products=data["products"],
            conditions=conditions,
            reaction_rate=data.get("reaction_rate"),
            yield_percentage=data.get("yield_percentage"),
            agents=data.get("agents"),
            source=data.get("source"),
        )

    def to_reaction_smiles(self) -> str:
        """Convert to reaction SMILES notation."""
        reactants_str = ".".join(self.reactants)
        products_str = ".".join(self.products)
        agents_str = ".".join(self.agents) if self.agents else ""
        return f"{reactants_str}>{agents_str}>{products_str}"


class ReactionDataset:
    """Dataset handler for chemical reactions with SMILES support."""

    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize reaction dataset.

        Args:
            dataset_path: Path to existing dataset file (JSON or CSV)
        """
        self.reactions: List[ChemicalReaction] = []
        self.smiles_parser = SMILESParser()
        self.reaction_parser = ReactionSMILES()

        if dataset_path and dataset_path.exists():
            self.load_dataset(dataset_path)

    def load_dataset(self, filepath: Path) -> None:
        """Load reactions from file.

        Args:
            filepath: Path to dataset file (.json or .csv)
        """
        suffix = filepath.suffix.lower()

        if suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
                self.reactions = [
                    ChemicalReaction.from_dict(r) for r in data["reactions"]
                ]
        elif suffix == ".csv":
            df = pd.read_csv(filepath)
            self.reactions = self._dataframe_to_reactions(df)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        print(f"Loaded {len(self.reactions)} reactions from {filepath}")

    def save_dataset(self, filepath: Path) -> None:
        """Save reactions to file.

        Args:
            filepath: Path to save dataset (.json or .csv)
        """
        suffix = filepath.suffix.lower()

        if suffix == ".json":
            data = {"reactions": [r.to_dict() for r in self.reactions]}
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        elif suffix == ".csv":
            df = self._reactions_to_dataframe()
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        print(f"Saved {len(self.reactions)} reactions to {filepath}")

    def add_reaction(self, reaction: ChemicalReaction) -> None:
        """Add a reaction to the dataset."""
        self.reactions.append(reaction)

    def filter_by_conditions(
        self,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        catalyst: Optional[str] = None,
    ) -> "ReactionDataset":
        """Filter reactions by conditions.

        Args:
            min_temp: Minimum temperature filter
            max_temp: Maximum temperature filter
            catalyst: Catalyst name filter

        Returns:
            New ReactionDataset with filtered reactions
        """
        filtered = ReactionDataset()

        for reaction in self.reactions:
            # Temperature filter
            if min_temp is not None and (
                reaction.conditions.temperature is None
                or reaction.conditions.temperature < min_temp
            ):
                continue
            if max_temp is not None and (
                reaction.conditions.temperature is None
                or reaction.conditions.temperature > max_temp
            ):
                continue

            # Catalyst filter
            if catalyst is not None and reaction.conditions.catalyst != catalyst:
                continue

            filtered.add_reaction(reaction)

        return filtered

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        temps = [
            r.conditions.temperature
            for r in self.reactions
            if r.conditions.temperature is not None
        ]
        rates = [
            r.reaction_rate
            for r in self.reactions
            if r.reaction_rate is not None
        ]

        stats = {
            "num_reactions": len(self.reactions),
            "temperature_stats": {
                "mean": np.mean(temps) if temps else None,
                "std": np.std(temps) if temps else None,
                "min": min(temps) if temps else None,
                "max": max(temps) if temps else None,
            },
            "rate_stats": {
                "mean": np.mean(rates) if rates else None,
                "std": np.std(rates) if rates else None,
                "min": min(rates) if rates else None,
                "max": max(rates) if rates else None,
            },
        }

        return stats

    def _dataframe_to_reactions(self, df: pd.DataFrame) -> List[ChemicalReaction]:
        """Convert DataFrame to list of reactions."""
        reactions = []

        for idx, row in df.iterrows():
            # Parse reaction SMILES if available
            if "reaction_smiles" in row and pd.notna(row["reaction_smiles"]):
                parsed = self.reaction_parser.parse_reaction(row["reaction_smiles"])
                reactants = parsed["reactants"]
                products = parsed["products"]
                agents = parsed["agents"] if parsed["agents"] else None
            else:
                reactants = [row.get("reactants", "")]
                products = [row.get("products", "")]
                agents = None

            conditions = ReactionConditions(
                temperature=row.get("temperature"),
                solvent=row.get("solvent"),
                catalyst=row.get("catalyst"),
            )

            reaction = ChemicalReaction(
                reaction_id=str(idx),
                reactants=reactants,
                products=products,
                conditions=conditions,
                reaction_rate=row.get("reaction_rate"),
                yield_percentage=row.get("yield"),
                agents=agents,
                source=row.get("source", "unknown"),
            )

            reactions.append(reaction)

        return reactions

    def _reactions_to_dataframe(self) -> pd.DataFrame:
        """Convert reactions to DataFrame."""
        data = []

        for reaction in self.reactions:
            row = {
                "reaction_id": reaction.reaction_id,
                "reaction_smiles": reaction.to_reaction_smiles(),
                "num_reactants": len(reaction.reactants),
                "num_products": len(reaction.products),
                "temperature": reaction.conditions.temperature,
                "solvent": reaction.conditions.solvent,
                "catalyst": reaction.conditions.catalyst,
                "reaction_rate": reaction.reaction_rate,
                "yield": reaction.yield_percentage,
                "source": reaction.source,
            }
            data.append(row)

        return pd.DataFrame(data)
