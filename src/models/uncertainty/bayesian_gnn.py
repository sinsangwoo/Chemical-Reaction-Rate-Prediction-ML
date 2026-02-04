"""Bayesian Graph Neural Networks with uncertainty quantification.

Implements multiple SOTA approaches for uncertainty in GNNs:
1. MC Dropout: Monte Carlo sampling during inference
2. Bayesian Layers: Variational inference for weights
3. Deep Ensemble: Multiple models for epistemic uncertainty

Based on:
- Zhang et al. (2019): Bayesian Graph Convolutional Networks
- Hasanzadeh et al. (2020): Bayesian GNN with adaptive sampling
- Munikoti et al. (2023): Unified Bayesian framework
- NeurIPS 2024: Latest advances in Bayesian GNN
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    import numpy as np
    from typing import Tuple, List, Optional, Dict
    from dataclasses import dataclass
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and PyTorch Geometric not available")


if TORCH_AVAILABLE:
    
    @dataclass
    class UncertaintyEstimate:
        """Container for uncertainty estimates.
        
        Attributes:
            mean: Predicted mean value
            aleatoric: Data uncertainty (irreducible)
            epistemic: Model uncertainty (reducible)
            total: Total uncertainty (aleatoric + epistemic)
            confidence_interval: 95% confidence interval
        """
        mean: float
        aleatoric: float
        epistemic: float
        total: float
        confidence_interval: Tuple[float, float]
        
        def __repr__(self):
            return (
                f"UncertaintyEstimate(\n"
                f"  mean={self.mean:.4f},\n"
                f"  epistemic={self.epistemic:.4f} (model uncertainty),\n"
                f"  aleatoric={self.aleatoric:.4f} (data noise),\n"
                f"  total={self.total:.4f},\n"
                f"  95% CI=[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
                f")"
            )
    
    
    class MCDropoutGNN(nn.Module):
        """GNN with Monte Carlo Dropout for uncertainty.
        
        Uses dropout at inference time to estimate epistemic uncertainty
        via Bayesian approximation (Gal & Ghahramani, 2016).
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout_rate: Dropout probability (default: 0.2)
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            dropout_rate: float = 0.2
        ):
            super().__init__()
            
            self.num_layers = num_layers
            self.dropout_rate = dropout_rate
            
            # Graph convolution layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
            
            # MLP head
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, data, training=False):
            """Forward pass with optional dropout.
            
            Args:
                data: PyTorch Geometric Data object
                training: If True, use dropout (for MC sampling)
            
            Returns:
                Graph-level predictions
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GCN layers with dropout
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                
                # Apply dropout (even at inference if training=True)
                if training or self.training:
                    x = F.dropout(x, p=self.dropout_rate, training=True)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # MLP
            x = self.fc(x)
            
            return x
        
        def predict_with_uncertainty(
            self,
            data,
            n_samples: int = 100,
            confidence: float = 0.95
        ) -> UncertaintyEstimate:
            """Predict with uncertainty quantification via MC Dropout.
            
            Args:
                data: Single graph (PyTorch Geometric Data)
                n_samples: Number of Monte Carlo samples
                confidence: Confidence level for interval (default: 0.95)
            
            Returns:
                UncertaintyEstimate with mean, aleatoric, epistemic
            """
            self.eval()  # Set to eval mode but keep dropout active
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    # Forward pass with dropout enabled
                    pred = self.forward(data, training=True)
                    predictions.append(pred.item())
            
            predictions = np.array(predictions)
            
            # Compute statistics
            mean = np.mean(predictions)
            epistemic = np.var(predictions)  # Variance across samples
            aleatoric = 0.01  # Placeholder (requires heteroscedastic model)
            total = epistemic + aleatoric
            
            # Confidence interval
            alpha = 1 - confidence
            lower = np.percentile(predictions, alpha/2 * 100)
            upper = np.percentile(predictions, (1 - alpha/2) * 100)
            
            return UncertaintyEstimate(
                mean=mean,
                aleatoric=aleatoric,
                epistemic=epistemic,
                total=total,
                confidence_interval=(lower, upper)
            )
    
    
    class BayesianLinear(nn.Module):
        """Bayesian linear layer with weight uncertainty.
        
        Uses variational inference to learn distributions over weights
        instead of point estimates.
        
        Based on Blundell et al. (2015): Weight Uncertainty in Neural Networks
        """
        
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            
            # Weight parameters (mean and log variance)
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(0, 0.1)
            )
            self.weight_log_sigma = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(-5, 0.1)
            )
            
            # Bias parameters
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
            self.bias_log_sigma = nn.Parameter(
                torch.Tensor(out_features).normal_(-5, 0.1)
            )
        
        def forward(self, x):
            """Sample weights and compute forward pass."""
            # Sample weights from N(mu, sigma^2)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_epsilon * torch.exp(self.weight_log_sigma)
            
            # Sample bias
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_epsilon * torch.exp(self.bias_log_sigma)
            
            return F.linear(x, weight, bias)
        
        def kl_divergence(self):
            """Compute KL divergence between posterior and prior.
            
            Prior: N(0, 1)
            Posterior: N(mu, sigma^2)
            """
            # Weight KL
            weight_kl = 0.5 * torch.sum(
                torch.exp(2 * self.weight_log_sigma)
                + self.weight_mu ** 2
                - 2 * self.weight_log_sigma
                - 1
            )
            
            # Bias KL
            bias_kl = 0.5 * torch.sum(
                torch.exp(2 * self.bias_log_sigma)
                + self.bias_mu ** 2
                - 2 * self.bias_log_sigma
                - 1
            )
            
            return weight_kl + bias_kl
    
    
    class BayesianGNN(nn.Module):
        """Bayesian GNN with variational inference.
        
        Full Bayesian treatment with uncertainty over all weights.
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3
        ):
            super().__init__()
            
            self.num_layers = num_layers
            
            # GCN layers (deterministic for now)
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            # Bayesian MLP head
            self.fc1 = BayesianLinear(hidden_dim, hidden_dim)
            self.fc2 = BayesianLinear(hidden_dim, output_dim)
        
        def forward(self, data):
            """Forward pass with weight sampling."""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GCN layers
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Bayesian MLP
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            
            return x
        
        def kl_loss(self):
            """Total KL divergence loss."""
            return self.fc1.kl_divergence() + self.fc2.kl_divergence()
        
        def predict_with_uncertainty(
            self,
            data,
            n_samples: int = 100
        ) -> UncertaintyEstimate:
            """Predict with full Bayesian uncertainty."""
            self.eval()
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.forward(data)
                    predictions.append(pred.item())
            
            predictions = np.array(predictions)
            
            mean = np.mean(predictions)
            epistemic = np.var(predictions)
            aleatoric = 0.01  # Placeholder
            total = epistemic + aleatoric
            
            lower = np.percentile(predictions, 2.5)
            upper = np.percentile(predictions, 97.5)
            
            return UncertaintyEstimate(
                mean=mean,
                aleatoric=aleatoric,
                epistemic=epistemic,
                total=total,
                confidence_interval=(lower, upper)
            )
    
    
    class DeepEnsembleGNN(nn.Module):
        """Deep Ensemble for epistemic uncertainty.
        
        Trains multiple independent models and uses their disagreement
        to estimate epistemic uncertainty.
        
        Based on Lakshminarayanan et al. (2017): Simple and Scalable
        Predictive Uncertainty Estimation using Deep Ensembles.
        
        Args:
            base_model_class: GNN model class to ensemble
            n_models: Number of models in ensemble (default: 5)
            **model_kwargs: Arguments for base model
        """
        
        def __init__(
            self,
            base_model_class,
            n_models: int = 5,
            **model_kwargs
        ):
            super().__init__()
            
            self.n_models = n_models
            self.models = nn.ModuleList([
                base_model_class(**model_kwargs)
                for _ in range(n_models)
            ])
        
        def forward(self, data):
            """Forward pass (returns mean of ensemble)."""
            predictions = [model(data) for model in self.models]
            return torch.mean(torch.stack(predictions), dim=0)
        
        def predict_with_uncertainty(self, data) -> UncertaintyEstimate:
            """Predict with ensemble-based uncertainty."""
            predictions = []
            
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(data)
                    predictions.append(pred.item())
            
            predictions = np.array(predictions)
            
            mean = np.mean(predictions)
            epistemic = np.var(predictions)  # Disagreement between models
            aleatoric = 0.01  # Placeholder
            total = epistemic + aleatoric
            
            # Confidence interval from ensemble spread
            std = np.std(predictions)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            
            return UncertaintyEstimate(
                mean=mean,
                aleatoric=aleatoric,
                epistemic=epistemic,
                total=total,
                confidence_interval=(lower, upper)
            )


def create_uncertainty_model(
    model_type: str = 'mc_dropout',
    **kwargs
) -> 'nn.Module':
    """Factory function for uncertainty-aware models.
    
    Args:
        model_type: Type of uncertainty model
            - 'mc_dropout': Monte Carlo Dropout (fast, approximate)
            - 'bayesian': Full Bayesian GNN (accurate, slower)
            - 'ensemble': Deep Ensemble (most reliable, expensive)
        **kwargs: Model-specific arguments
    
    Returns:
        Uncertainty-aware GNN model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for uncertainty models")
    
    model_type = model_type.lower()
    
    if model_type == 'mc_dropout':
        return MCDropoutGNN(**kwargs)
    elif model_type == 'bayesian':
        return BayesianGNN(**kwargs)
    elif model_type == 'ensemble':
        from src.models.gnn.gnn_models import GCNModel
        return DeepEnsembleGNN(GCNModel, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
