"""Loader for USPTO chemical reaction dataset.

USPTO (United States Patent and Trademark Office) reaction dataset contains
millions of chemical reactions extracted from patents.

Key datasets:
- USPTO-50K: 50,000 curated reactions
- USPTO-MIT: ~480K reactions
- USPTO-FULL: 1M+ reactions
- USPTO-LLM: 247K reactions with rich conditions (2024)

References:
- Lowe, D. (2017). Chemical reactions from US patents (1976-Sep2016)
- ORDerly: https://pmc.ncbi.nlm.nih.gov/articles/PMC11094788/
- USPTO-LLM: https://zenodo.org/records/14396156
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import requests
import gzip
import json
from tqdm import tqdm

from .reaction_dataset import (
    ReactionDataset,
    ChemicalReaction,
    ReactionConditions,
)


class USPTOLoader:
    """Load and process USPTO reaction datasets."""

    # Official download URLs
    URLS = {
        "uspto_50k_sample": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/USPTO/data/raw_train.csv",
        # Add more URLs as needed
    }

    def __init__(self, cache_dir: Path = Path("data/raw/uspto")):
        """Initialize USPTO loader.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(
        self, dataset_name: str = "uspto_50k_sample", force: bool = False
    ) -> Path:
        """Download USPTO dataset.

        Args:
            dataset_name: Name of dataset to download
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file
        """
        if dataset_name not in self.URLS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.URLS.keys())}"
            )

        url = self.URLS[dataset_name]
        filename = url.split("/")[-1]
        filepath = self.cache_dir / filename

        if filepath.exists() and not force:
            print(f"Using cached file: {filepath}")
            return filepath

        print(f"Downloading {dataset_name} from {url}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=filename,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"Downloaded to {filepath}")
            return filepath

        except Exception as e:
            print(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            raise

    def load_from_csv(
        self,
        filepath: Path,
        max_reactions: Optional[int] = None,
        sample_fraction: float = 1.0,
    ) -> ReactionDataset:
        """Load USPTO reactions from CSV file.

        Args:
            filepath: Path to CSV file
            max_reactions: Maximum number of reactions to load
            sample_fraction: Fraction of data to randomly sample (0.0-1.0)

        Returns:
            ReactionDataset with loaded reactions
        """
        print(f"Loading USPTO reactions from {filepath}...")

        df = pd.read_csv(filepath)

        # Sample if requested
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)

        # Limit number of reactions
        if max_reactions:
            df = df.head(max_reactions)

        dataset = ReactionDataset()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                reaction = self._parse_uspto_row(row, idx)
                if reaction:
                    dataset.add_reaction(reaction)
            except Exception as e:
                print(f"Failed to parse reaction {idx}: {e}")
                continue

        print(f"Successfully loaded {len(dataset.reactions)} reactions")
        return dataset

    def _parse_uspto_row(self, row: pd.Series, idx: int) -> Optional[ChemicalReaction]:
        """Parse a USPTO CSV row into a ChemicalReaction.

        Args:
            row: Pandas Series from USPTO CSV
            idx: Row index

        Returns:
            ChemicalReaction or None if parsing fails
        """
        # USPTO CSV columns vary by dataset version
        # Common formats:
        # - 'reactants>reagents>production' (reaction SMILES)
        # - separate 'reactants', 'products' columns
        # - 'CanonicalizedReaction' field

        reaction_smiles = None

        # Try different column names
        for col in ["rxn_smiles", "reaction", "ReactionSmiles", "CanonicalizedReaction"]:
            if col in row.index and pd.notna(row[col]):
                reaction_smiles = str(row[col])
                break

        if not reaction_smiles:
            return None

        # Parse reaction SMILES
        from .smiles_parser import ReactionSMILES

        parser = ReactionSMILES()

        try:
            parsed = parser.parse_reaction(reaction_smiles)
        except Exception:
            return None

        # Extract conditions (if available)
        conditions = ReactionConditions(
            temperature=row.get("temperature"),
            solvent=row.get("solvent"),
            catalyst=row.get("catalyst"),
        )

        # Create reaction
        reaction = ChemicalReaction(
            reaction_id=f"uspto_{idx}",
            reactants=parsed["reactants"],
            products=parsed["products"],
            agents=parsed["agents"] if parsed["agents"] else None,
            conditions=conditions,
            yield_percentage=row.get("yield"),
            source="USPTO",
        )

        return reaction

    def create_synthetic_dataset(
        self, num_reactions: int = 1000
    ) -> ReactionDataset:
        """Create synthetic USPTO-like dataset for testing.

        Args:
            num_reactions: Number of reactions to generate

        Returns:
            ReactionDataset with synthetic reactions
        """
        print(f"Generating {num_reactions} synthetic USPTO-style reactions...")

        # Common reaction SMILES patterns
        example_reactions = [
            "CCO.CC(=O)O>>CCOC(=O)C",  # Esterification
            "c1ccccc1Br.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1",  # Suzuki
            "CC(C)=O.NCC>>CC(C)=NCC",  # Imine formation
            "CCCl.N>>CCNC",  # Nucleophilic substitution
            "c1ccccc1.Br2>>c1ccc(Br)cc1",  # Aromatic bromination
        ]

        dataset = ReactionDataset()

        for i in range(num_reactions):
            # Randomly select a reaction template
            rxn_smiles = np.random.choice(example_reactions)

            # Random conditions
            conditions = ReactionConditions(
                temperature=np.random.uniform(20, 120),
                solvent=np.random.choice(["DCM", "THF", "toluene", "water", None]),
                catalyst=np.random.choice(["Pd(PPh3)4", "H2SO4", "NaOH", None]),
            )

            # Parse reaction
            from .smiles_parser import ReactionSMILES

            parser = ReactionSMILES()
            parsed = parser.parse_reaction(rxn_smiles)

            # Create reaction
            reaction = ChemicalReaction(
                reaction_id=f"synthetic_{i}",
                reactants=parsed["reactants"],
                products=parsed["products"],
                agents=parsed["agents"] if parsed["agents"] else None,
                conditions=conditions,
                yield_percentage=np.random.uniform(40, 95),
                source="synthetic",
            )

            dataset.add_reaction(reaction)

        print(f"Generated {len(dataset.reactions)} synthetic reactions")
        return dataset


if __name__ == "__main__":
    # Example usage
    loader = USPTOLoader()

    # Create synthetic dataset for testing
    dataset = loader.create_synthetic_dataset(num_reactions=100)

    # Save to file
    output_path = Path("data/processed/synthetic_uspto.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_dataset(output_path)

    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    print(json.dumps(stats, indent=2))
