"""Arrhenius equation utilities.

k(T) = A * exp(-Ea / RT)

All energies in kJ/mol, temperatures in K.
"""

from __future__ import annotations

import math
from typing import List, Tuple

R = 8.314e-3  # kJ / (mol · K)


def arrhenius_rate(
    frequency_factor: float,
    activation_energy: float,
    temperature: float,
) -> float:
    """Compute the Arrhenius rate constant k(T).

    Parameters
    ----------
    frequency_factor : float
        Pre-exponential factor A  (s^-1 for unimolecular, L/mol/s for
        bimolecular).
    activation_energy : float
        Activation energy Ea in kJ/mol.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Rate constant k at the given temperature.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature} K")
    return frequency_factor * math.exp(-activation_energy / (R * temperature))


def activation_energy_from_rates(
    k1: float, T1: float, k2: float, T2: float
) -> float:
    """Back-calculate Ea from two rate constants at two temperatures.

    Uses the two-temperature Arrhenius form:
        Ea = R * ln(k2/k1) / (1/T1 - 1/T2)

    Parameters
    ----------
    k1, k2 : float
        Rate constants at temperatures T1 and T2 respectively.
    T1, T2 : float
        Temperatures in Kelvin.

    Returns
    -------
    float
        Estimated Ea in kJ/mol.
    """
    if k1 <= 0 or k2 <= 0:
        raise ValueError("Rate constants must be positive")
    if T1 == T2:
        raise ValueError("Temperatures must differ")
    return R * math.log(k2 / k1) / (1.0 / T1 - 1.0 / T2)


def rate_vs_temperature(
    frequency_factor: float,
    activation_energy: float,
    temperatures: List[float],
) -> List[Tuple[float, float]]:
    """Return a list of (T, k(T)) pairs over a range of temperatures."""
    return [
        (T, arrhenius_rate(frequency_factor, activation_energy, T))
        for T in temperatures
    ]
