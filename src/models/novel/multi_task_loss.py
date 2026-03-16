"""Multi-task loss for simultaneous Ea, ln(A), and k prediction.

Combines three objectives with learnable uncertainty weights following
the approach of Kendall et al. (2018) "Multi-Task Learning Using
Uncertainty to Weigh Losses in Deep Learning".

Usage
-----
    criterion = MultiTaskKineticLoss()
    loss = criterion(ea_pred, ea_true, log_a_pred, log_a_true, k_pred, k_true)
    loss.backward()
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskKineticLoss(nn.Module):
    """Homoscedastic uncertainty-weighted multi-task loss.

    Learns log-variance parameters ``log_sigma_ea``, ``log_sigma_a``,
    and ``log_sigma_k`` which adaptively balance the three losses during
    training.

    Parameters
    ----------
    ea_weight : float
        Initial weight for the Ea loss (overridden by learned sigma).
    a_weight : float
        Initial weight for the ln(A) loss.
    k_weight : float
        Initial weight for the rate-constant loss.
    reduction : str
        ``'mean'`` (default) or ``'sum'``.
    """

    def __init__(
        self,
        ea_weight: float = 1.0,
        a_weight: float = 1.0,
        k_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        # Learnable log-variance parameters (initialised to log(weight))
        self.log_sigma_ea = nn.Parameter(torch.tensor(math.log(ea_weight)))
        self.log_sigma_a = nn.Parameter(torch.tensor(math.log(a_weight)))
        self.log_sigma_k = nn.Parameter(torch.tensor(math.log(k_weight)))

    def _weighted(self, loss: Tensor, log_sigma: Tensor) -> Tensor:
        """Apply homoscedastic uncertainty weighting."""
        # L_weighted = L / (2 * exp(log_sigma)^2) + log_sigma
        return loss / (2.0 * torch.exp(2.0 * log_sigma)) + log_sigma

    def forward(
        self,
        ea_pred: Tensor,
        ea_true: Tensor,
        log_a_pred: Tensor,
        log_a_true: Tensor,
        k_pred: Tensor,
        k_true: Tensor,
    ) -> Tensor:
        """Compute total uncertainty-weighted loss.

        Parameters
        ----------
        ea_pred, ea_true : Tensor
            Predicted / ground-truth activation energies (kJ/mol).
        log_a_pred, log_a_true : Tensor
            Predicted / ground-truth ln(A) values.
        k_pred, k_true : Tensor
            Predicted / ground-truth rate constants.

        Returns
        -------
        Tensor
            Scalar total loss.
        """
        mse = nn.functional.mse_loss

        loss_ea = mse(ea_pred, ea_true, reduction=self.reduction)
        loss_a = mse(log_a_pred, log_a_true, reduction=self.reduction)
        # Rate constants span many orders of magnitude — use log-MSE
        loss_k = mse(
            torch.log(k_pred.clamp(min=1e-30)),
            torch.log(k_true.clamp(min=1e-30)),
            reduction=self.reduction,
        )

        total = (
            self._weighted(loss_ea, self.log_sigma_ea)
            + self._weighted(loss_a, self.log_sigma_a)
            + self._weighted(loss_k, self.log_sigma_k)
        )
        return total

    @property
    def effective_weights(self) -> dict:
        """Current effective weights for logging / monitoring."""
        return {
            "ea": float(torch.exp(2.0 * self.log_sigma_ea).item()),
            "a": float(torch.exp(2.0 * self.log_sigma_a).item()),
            "k": float(torch.exp(2.0 * self.log_sigma_k).item()),
        }
