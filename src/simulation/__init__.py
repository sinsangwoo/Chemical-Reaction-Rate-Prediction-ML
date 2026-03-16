"""Kinetic simulation sub-package.

Provides the Kinetic Monte Carlo solver and Arrhenius utility functions
for time-resolved concentration profiles.
"""

from .kmc_solver import KMCSolver, KMCResult
from .arrhenius_utils import arrhenius_rate, activation_energy_from_rates

__all__ = [
    "KMCSolver",
    "KMCResult",
    "arrhenius_rate",
    "activation_energy_from_rates",
]
