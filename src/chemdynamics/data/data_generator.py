"""Generate synthetic chemical reaction data based on Arrhenius equation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class ArrheniusDataGenerator:
    """Generate chemical reaction rate data using Arrhenius equation."""

    def __init__(
        self,
        frequency_factor: float = 1e5,
        activation_energy: float = 40000,
        activation_energy_catalyst: float = 25000,
        gas_constant: float = 8.314,
        random_seed: int = 42,
    ):
        """Initialize the data generator.

        Args:
            frequency_factor: Pre-exponential factor (A) in Arrhenius equation
            activation_energy: Activation energy without catalyst (J/mol)
            activation_energy_catalyst: Activation energy with catalyst (J/mol)
            gas_constant: Universal gas constant (J/mol·K)
            random_seed: Random seed for reproducibility
        """
        self.A = frequency_factor
        self.Ea = activation_energy
        self.Ea_cat = activation_energy_catalyst
        self.R = gas_constant
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_data(
        self,
        num_samples: int = 300,
        temp_range: tuple[float, float] = (10, 90),
        conc_range: tuple[float, float] = (0.1, 2.5),
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic reaction rate data.

        Args:
            num_samples: Number of data points to generate
            temp_range: Temperature range in Celsius (min, max)
            conc_range: Concentration range in mol/L (min, max)
            noise_level: Relative noise level (std as fraction of mean)

        Returns:
            DataFrame with columns: temperature, concentration, catalyst, reaction_rate
        """
        # Generate independent variables
        temperatures_c = np.random.uniform(*temp_range, num_samples)
        temperatures_k = temperatures_c + 273.15
        concentrations = np.random.uniform(*conc_range, num_samples)
        catalysts = np.random.randint(0, 2, num_samples)

        # Calculate rate constants using Arrhenius equation
        k_no_cat = self.A * np.exp(-self.Ea / (self.R * temperatures_k))
        k_cat = self.A * np.exp(-self.Ea_cat / (self.R * temperatures_k))

        # Apply rate law: rate = k * [C]
        base_rate = np.where(catalysts == 1, k_cat, k_no_cat) * concentrations

        # Add measurement noise
        noise = np.random.normal(0, np.mean(base_rate) * noise_level, num_samples)
        reaction_rates = base_rate + noise
        reaction_rates = np.maximum(reaction_rates, 0)  # Rates cannot be negative

        # Create DataFrame
        data = pd.DataFrame(
            {
                "temperature_C": temperatures_c,
                "concentration_mol_L": concentrations,
                "catalyst": catalysts,
                "reaction_rate_mol_L_s": reaction_rates,
            }
        )

        return data

    def save_data(self, data: pd.DataFrame, filepath: Path) -> None:
        """Save generated data to CSV file.

        Args:
            data: DataFrame to save
            filepath: Path to save the CSV file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


if __name__ == "__main__":
    # Generate and save data
    generator = ArrheniusDataGenerator()
    data = generator.generate_data(num_samples=300)
    print(data.head())
    print(f"\nGenerated {len(data)} samples")
    print(f"\nData statistics:\n{data.describe()}")
