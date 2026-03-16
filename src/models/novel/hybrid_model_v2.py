"""HybridGNN v2 — Multi-output Kinetic Predictor.

Extends the original HybridGNN with two dedicated output heads that
predict per-elementary-step kinetic parameters:

* **Ea head** — activation energy (kJ/mol) for each edge / reaction step.
* **A head**  — natural-log of the pre-exponential factor ln(A) for
  numerical stability; exponentiated before returning.

Bayesian uncertainty is produced by applying MC Dropout at inference time
over *both* shared and per-head layers, enabling the KMC Solver to
propagate uncertainty through the Gillespie trajectories.

Backward-compatible: the ``forward()`` method still returns a scalar rate
constant by default; set ``return_kinetic_params=True`` to get
``(k, Ea, A, Ea_std, A_std)``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class KineticOutputHead(nn.Module):
    """Shared MLP head used for Ea and ln(A) predictions.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (from shared GNN backbone).
    hidden_dim : int
        Hidden layer size.
    dropout : float
        Dropout probability (applied at inference for MC Dropout).
    min_val : float
        Hard lower bound applied via Softplus shift.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        min_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        """Return positive-valued prediction shifted by ``min_val``."""
        return self.softplus(self.net(x).squeeze(-1)) + self.min_val


class MultiStepHybridGNN(nn.Module):
    """Physics-informed GNN with per-step Ea and A prediction heads.

    Architecture
    ------------
    1. Shared embedding backbone (3-layer MLP on molecular features).
    2. Ea head  — ``KineticOutputHead`` predicting activation energy.
    3. ln(A) head — ``KineticOutputHead`` predicting log pre-exponential.
    4. Rate head — combines Ea + A via Arrhenius to predict k(T).

    Parameters
    ----------
    node_features : int
        Dimensionality of the molecular input feature vector.
    hidden_dim : int
        Width of the shared backbone layers.
    dropout : float
        Dropout probability used in every sub-network.
    n_mc_samples : int
        Number of forward passes for MC Dropout uncertainty estimation.
    """

    R: float = 8.314e-3  # kJ / (mol · K)

    def __init__(
        self,
        node_features: int = 37,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        n_mc_samples: int = 50,
    ) -> None:
        super().__init__()
        self.n_mc_samples = n_mc_samples

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Kinetic output heads
        self.ea_head = KineticOutputHead(
            in_dim=hidden_dim // 2,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
            min_val=1.0,    # Ea ≥ 1 kJ/mol
        )
        self.log_a_head = KineticOutputHead(
            in_dim=hidden_dim // 2,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
            min_val=0.0,    # ln(A) ≥ 0 ⇒ A ≥ 1 s^-1
        )

        # Scalar rate-constant output head (legacy compatibility)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        temperature: Optional[float] = None,
        return_kinetic_params: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, node_features)``.
        temperature : float, optional
            If provided, the Arrhenius rate is computed using this temperature
            and returned as the primary scalar output.
        return_kinetic_params : bool
            When True, return ``(k, Ea, A)`` instead of just ``k``.

        Returns
        -------
        Tensor or tuple
            ``k`` of shape ``(batch,)`` when ``return_kinetic_params=False``.
            ``(k, Ea, A)`` each of shape ``(batch,)`` otherwise.
        """
        h = self.backbone(x)
        ea = self.ea_head(h)          # kJ/mol
        log_a = self.log_a_head(h)    # dimensionless (ln scale)
        a = torch.exp(log_a)          # s^-1

        if temperature is not None:
            k = a * torch.exp(-ea / (self.R * temperature))
        else:
            k = self.rate_head(h).squeeze(-1)

        if return_kinetic_params:
            return k, ea, a
        return k

    # ------------------------------------------------------------------
    # MC Dropout uncertainty
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        x: Tensor,
        temperature: float = 298.15,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Estimate Ea and A with Bayesian uncertainty via MC Dropout.

        Activates dropout at inference and runs ``n_samples`` stochastic
        forward passes, then returns the mean and standard deviation.

        Parameters
        ----------
        x : Tensor
            Molecular feature tensor, shape ``(batch, node_features)``.
        temperature : float
            Temperature in Kelvin for Arrhenius rate computation.
        n_samples : int, optional
            Override the constructor-level ``n_mc_samples``.

        Returns
        -------
        tuple of Tensor
            ``(k_mean, ea_mean, a_mean, ea_std, a_std)``
            each of shape ``(batch,)``.
        """
        n = n_samples or self.n_mc_samples
        self.train()  # enable dropout

        k_samples, ea_samples, a_samples = [], [], []
        with torch.no_grad():
            for _ in range(n):
                k, ea, a = self.forward(
                    x, temperature=temperature, return_kinetic_params=True
                )
                k_samples.append(k)
                ea_samples.append(ea)
                a_samples.append(a)

        self.eval()

        k_stack = torch.stack(k_samples)    # (n, batch)
        ea_stack = torch.stack(ea_samples)
        a_stack = torch.stack(a_samples)

        return (
            k_stack.mean(0),
            ea_stack.mean(0),
            a_stack.mean(0),
            ea_stack.std(0),
            a_stack.std(0),
        )

    # ------------------------------------------------------------------
    # Convenience: export for RPG integration
    # ------------------------------------------------------------------

    def predict_kinetic_params_for_rpg(
        self,
        x: Tensor,
        temperature: float = 298.15,
    ) -> dict:
        """Return a dict ready to be passed to ``ReactionPathGraph.build_from_smiles``.

        Returns
        -------
        dict with keys:
            ``ea_values`` — list[float], kJ/mol per step
            ``a_values``  — list[float], s^-1 per step
            ``ea_uncertainties`` — list[float], 1-sigma kJ/mol per step
        """
        k_mean, ea_mean, a_mean, ea_std, a_std = self.predict_with_uncertainty(
            x, temperature=temperature
        )
        return {
            "ea_values": ea_mean.tolist(),
            "a_values": a_mean.tolist(),
            "ea_uncertainties": ea_std.tolist(),
        }
