"""Dataset class for multi-step reaction network training data.

Supports two data formats:

1. **CSV format** — columns: ``smiles``, ``ea_kj_mol``, ``log_a``,
   ``temperature``, ``rate_constant`` (all existing data).
2. **Network format** — JSON records with keys: ``reactant_smiles``,
   ``product_smiles``, ``intermediate_smiles`` (list), ``ea_values`` (list),
   ``a_values`` (list), ``temperature``.

The dataset yields ``(features, ea, log_a, k)`` tuples suitable for
training ``MultiStepHybridGNN`` with a multi-task loss.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ReactionNetworkSample:
    """One multi-step reaction record."""
    reactant_smiles: List[str]
    product_smiles: List[str]
    intermediate_smiles: List[str]
    ea_values: List[float]          # kJ/mol per elementary step
    a_values: List[float]           # s^-1 per step
    temperature: float              # K
    rate_constant: Optional[float] = None  # observed overall k, if available

    @property
    def log_a_values(self) -> List[float]:
        return [math.log(max(a, 1e-30)) for a in self.a_values]


class ReactionNetworkDataset:
    """In-memory dataset for multi-step reaction networks.

    Parameters
    ----------
    records : list of ReactionNetworkSample
        Pre-loaded samples.
    feature_dim : int
        Expected feature vector dimensionality (for validation).
    """

    def __init__(
        self,
        records: Optional[List[ReactionNetworkSample]] = None,
        feature_dim: int = 37,
    ) -> None:
        self.records: List[ReactionNetworkSample] = records or []
        self.feature_dim = feature_dim

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str, feature_dim: int = 37) -> "ReactionNetworkDataset":
        """Load a JSON file of network records.

        Expected JSON structure::

            [
              {
                "reactant_smiles": ["CCO"],
                "product_smiles": ["CC=O"],
                "intermediate_smiles": ["CC[OH2+]"],
                "ea_values": [80.0, 60.0],
                "a_values": [1e13, 1e12],
                "temperature": 500.0
              },
              ...
            ]
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        records = [
            ReactionNetworkSample(
                reactant_smiles=r.get("reactant_smiles", []),
                product_smiles=r.get("product_smiles", []),
                intermediate_smiles=r.get("intermediate_smiles", []),
                ea_values=r.get("ea_values", []),
                a_values=r.get("a_values", []),
                temperature=r.get("temperature", 298.15),
                rate_constant=r.get("rate_constant"),
            )
            for r in raw
        ]
        return cls(records=records, feature_dim=feature_dim)

    @classmethod
    def synthetic(
        cls,
        n_samples: int = 100,
        seed: int = 42,
        feature_dim: int = 37,
    ) -> "ReactionNetworkDataset":
        """Generate a synthetic dataset for unit tests and demos.

        Each sample is a 2-step reaction A → B → C with randomised
        Ea values drawn from N(70, 15²) kJ/mol and temperatures
        uniformly sampled from [300, 800] K.
        """
        import random
        rng = random.Random(seed)
        records = []
        species = ["A", "B", "C", "D", "E"]
        for i in range(n_samples):
            n_steps = rng.randint(1, 3)
            smiles = species[: n_steps + 1]
            ea = [max(10.0, rng.gauss(70.0, 15.0)) for _ in range(n_steps)]
            a = [10 ** rng.uniform(10, 14) for _ in range(n_steps)]
            T = rng.uniform(300.0, 800.0)
            records.append(
                ReactionNetworkSample(
                    reactant_smiles=[smiles[0]],
                    product_smiles=[smiles[-1]],
                    intermediate_smiles=smiles[1:-1],
                    ea_values=ea,
                    a_values=a,
                    temperature=T,
                )
            )
        return cls(records=records, feature_dim=feature_dim)

    # ------------------------------------------------------------------
    # Standard dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ReactionNetworkSample:
        return self.records[idx]

    def split(
        self, train_frac: float = 0.8, seed: int = 42
    ) -> Tuple["ReactionNetworkDataset", "ReactionNetworkDataset"]:
        """Random train / validation split."""
        import random
        rng = random.Random(seed)
        indices = list(range(len(self.records)))
        rng.shuffle(indices)
        cut = int(len(indices) * train_frac)
        train_recs = [self.records[i] for i in indices[:cut]]
        val_recs = [self.records[i] for i in indices[cut:]]
        return (
            ReactionNetworkDataset(train_recs, self.feature_dim),
            ReactionNetworkDataset(val_recs, self.feature_dim),
        )

    def summary(self) -> str:
        n = len(self.records)
        if n == 0:
            return "ReactionNetworkDataset (empty)"
        avg_steps = sum(len(r.ea_values) for r in self.records) / n
        return (
            f"ReactionNetworkDataset  samples={n}  "
            f"avg_steps={avg_steps:.1f}  "
            f"feature_dim={self.feature_dim}"
        )
