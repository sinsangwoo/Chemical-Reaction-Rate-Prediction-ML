"""Extract molecular features from SMILES for ML models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..data.smiles_parser import SMILESParser
from ..data.reaction_dataset import ChemicalReaction


@dataclass
class MolecularFingerprint:
    """Container for molecular fingerprint features."""

    # Composition features
    num_atoms: int
    num_carbons: int
    num_nitrogens: int
    num_oxygens: int
    num_halogens: int

    # Structural features
    num_rings: int
    num_branches: int
    has_aromatic: bool
    estimated_mw: float

    # Complexity metrics
    smiles_length: int
    atom_diversity: float  # Shannon entropy of atom types

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array(
            [
                self.num_atoms,
                self.num_carbons,
                self.num_nitrogens,
                self.num_oxygens,
                self.num_halogens,
                self.num_rings,
                self.num_branches,
                float(self.has_aromatic),
                self.estimated_mw,
                self.smiles_length,
                self.atom_diversity,
            ]
        )

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for ML."""
        return [
            "num_atoms",
            "num_carbons",
            "num_nitrogens",
            "num_oxygens",
            "num_halogens",
            "num_rings",
            "num_branches",
            "has_aromatic",
            "estimated_mw",
            "smiles_length",
            "atom_diversity",
        ]


class MolecularFeatureExtractor:
    """Extract features from molecular SMILES."""

    def __init__(self):
        """Initialize feature extractor."""
        self.parser = SMILESParser()

    def extract_fingerprint(self, smiles: str) -> MolecularFingerprint:
        """Extract molecular fingerprint from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            MolecularFingerprint with extracted features
        """
        features = self.parser.extract_features(smiles)
        atom_counts = features["atom_counts"]

        # Calculate atom diversity (Shannon entropy)
        total_atoms = sum(atom_counts.values())
        if total_atoms > 0:
            probs = np.array(list(atom_counts.values())) / total_atoms
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0.0

        return MolecularFingerprint(
            num_atoms=total_atoms,
            num_carbons=atom_counts.get("C", 0),
            num_nitrogens=atom_counts.get("N", 0),
            num_oxygens=atom_counts.get("O", 0),
            num_halogens=sum(
                atom_counts.get(x, 0) for x in ["F", "CL", "BR", "I"]
            ),
            num_rings=features["num_rings"],
            num_branches=features["num_branches"],
            has_aromatic=features["has_aromatic"],
            estimated_mw=features["estimated_mw"],
            smiles_length=features["length"],
            atom_diversity=entropy,
        )

    def extract_reaction_features(self, reaction: ChemicalReaction) -> pd.DataFrame:
        """Extract features from a complete reaction.

        Args:
            reaction: ChemicalReaction object

        Returns:
            DataFrame with reaction features
        """
        # Extract fingerprints for all reactants
        reactant_fps = [
            self.extract_fingerprint(r) for r in reaction.reactants
        ]

        # Extract fingerprints for all products
        product_fps = [self.extract_fingerprint(p) for p in reaction.products]

        # Aggregate reactant features (sum)
        reactant_features = np.sum(
            [fp.to_vector() for fp in reactant_fps], axis=0
        )

        # Aggregate product features (sum)
        product_features = np.sum(
            [fp.to_vector() for fp in product_fps], axis=0
        )

        # Delta features (product - reactant)
        delta_features = product_features - reactant_features

        # Condition features
        condition_features = np.array(
            [
                reaction.conditions.temperature or 25.0,  # Default room temp
                1.0
                if reaction.conditions.catalyst
                else 0.0,  # Binary catalyst flag
                len(reaction.reactants),
                len(reaction.products),
            ]
        )

        # Combine all features
        all_features = np.concatenate(
            [reactant_features, product_features, delta_features, condition_features]
        )

        # Create feature names
        fp_names = MolecularFingerprint.feature_names()
        feature_names = (
            [f"reactant_{n}" for n in fp_names]
            + [f"product_{n}" for n in fp_names]
            + [f"delta_{n}" for n in fp_names]
            + ["temperature", "has_catalyst", "num_reactants", "num_products"]
        )

        return pd.DataFrame([all_features], columns=feature_names)


class ReactionFeatureBuilder:
    """Build feature matrices for ML from reaction datasets."""

    def __init__(self):
        """Initialize feature builder."""
        self.extractor = MolecularFeatureExtractor()

    def build_features(
        self,
        reactions: List[ChemicalReaction],
        include_rates: bool = True,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Build feature matrix from reactions.

        Args:
            reactions: List of ChemicalReaction objects
            include_rates: Whether to extract target rates

        Returns:
            (features_df, rates_series) tuple
        """
        feature_dfs = []
        rates = []

        for reaction in reactions:
            try:
                features = self.extractor.extract_reaction_features(reaction)
                feature_dfs.append(features)

                if include_rates and reaction.reaction_rate is not None:
                    rates.append(reaction.reaction_rate)
                elif include_rates and reaction.yield_percentage is not None:
                    # Use yield as proxy for rate
                    rates.append(reaction.yield_percentage / 100.0)
                else:
                    rates.append(None)

            except Exception as e:
                print(f"Failed to extract features: {e}")
                continue

        if not feature_dfs:
            raise ValueError("No features could be extracted")

        features_df = pd.concat(feature_dfs, ignore_index=True)

        if include_rates:
            rates_series = pd.Series(rates, name="target")
            # Drop rows with missing rates
            valid_idx = rates_series.notna()
            features_df = features_df[valid_idx].reset_index(drop=True)
            rates_series = rates_series[valid_idx].reset_index(drop=True)
            return features_df, rates_series
        else:
            return features_df, None
