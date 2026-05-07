"""Uncertainty quantification modules for molecular property prediction.

This module implements state-of-the-art uncertainty quantification techniques
for Graph Neural Networks, based on recent research (2024-2025).

Key Methods:
- Bayesian GNN: Variational inference with Monte Carlo dropout
- Deep Ensemble: Multiple models with epistemic uncertainty
- MVE (Mean Variance Estimation): Aleatoric + Epistemic
- Evidential Deep Learning: Dirichlet distributions
- Conformal Prediction: Distribution-free guarantees

References:
- Zhang et al. (2019): Bayesian GCN
- Hasanzadeh et al. (2020): Bayesian GNN framework
- Munikoti et al. (2023): Unified Bayesian approach
- Cha et al. (2023): Conformal prediction for GNN
- Nature Comm (2025): UQ with GNN for molecular design
"""
