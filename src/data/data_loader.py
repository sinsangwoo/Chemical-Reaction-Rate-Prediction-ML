"""Data loading and preprocessing utilities."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np


class ReactionDataLoader:
    """Load and preprocess chemical reaction data."""

    def __init__(self, data_path: Path):
        """Initialize data loader.

        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.data: pd.DataFrame = None

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.

        Returns:
            Loaded DataFrame
        """
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} samples from {self.data_path}")
        return self.data

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Identify feature and target columns
        feature_cols = [col for col in self.data.columns if "rate" not in col.lower()]
        target_col = [col for col in self.data.columns if "rate" in col.lower()][0]

        X = self.data[feature_cols]
        y = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def get_statistics(self) -> dict:
        """Get basic statistics about the dataset.

        Returns:
            Dictionary of statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {
            "num_samples": len(self.data),
            "num_features": len(self.data.columns) - 1,
            "feature_names": [col for col in self.data.columns if "rate" not in col.lower()],
            "target_name": [col for col in self.data.columns if "rate" in col.lower()][0],
            "catalyst_distribution": self.data["catalyst"].value_counts().to_dict()
            if "catalyst" in self.data.columns
            else None,
        }

        return stats
