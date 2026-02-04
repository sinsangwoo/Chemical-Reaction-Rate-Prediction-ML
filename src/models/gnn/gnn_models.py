"""Graph Neural Network models using PyTorch Geometric.

Implements state-of-the-art GNN architectures for molecular property prediction:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)  
- MPNN (Message Passing Neural Network)
- GIN (Graph Isomorphism Network)

References:
- Kipf & Welling (2017): Semi-Supervised Classification with GCNs
- Veličković et al. (2018): Graph Attention Networks
- Gilmer et al. (2017): Neural Message Passing for Quantum Chemistry
- Xu et al. (2019): How Powerful are Graph Neural Networks?
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import (
        GCNConv,
        GATConv,
        GINConv,
        global_mean_pool,
        global_max_pool,
        global_add_pool,
    )
    from torch_geometric.data import Data, Batch
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and PyTorch Geometric not available.")
    print("Install with: pip install torch torch-geometric")


if TORCH_AVAILABLE:
    
    class GCNModel(nn.Module):
        """Graph Convolutional Network for molecular property prediction.
        
        Architecture:
            Input → GCN layers → Global pooling → MLP → Output
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for regression)
            num_layers: Number of GCN layers
            dropout: Dropout probability
            pooling: Graph pooling method ('mean', 'max', 'add')
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            dropout: float = 0.2,
            pooling: str = 'mean'
        ):
            super().__init__()
            
            self.num_layers = num_layers
            self.dropout = dropout
            self.pooling = pooling
            
            # Graph convolutional layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
            
            # MLP for graph-level prediction
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, data):
            """Forward pass.
            
            Args:
                data: PyTorch Geometric Data object
                    - x: Node features [num_nodes, input_dim]
                    - edge_index: Edge connectivity [2, num_edges]
                    - batch: Batch assignment [num_nodes]
            
            Returns:
                Graph-level predictions [batch_size, output_dim]
            """
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GCN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            x = self._global_pool(x, batch)
            
            # MLP
            x = self.mlp(x)
            
            return x
        
        def _global_pool(self, x, batch):
            """Apply global pooling."""
            if self.pooling == 'mean':
                return global_mean_pool(x, batch)
            elif self.pooling == 'max':
                return global_max_pool(x, batch)
            elif self.pooling == 'add':
                return global_add_pool(x, batch)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
    
    
    class GATModel(nn.Module):
        """Graph Attention Network for molecular property prediction.
        
        Uses multi-head attention to learn edge importance.
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.num_layers = num_layers
            self.dropout = dropout
            
            # GAT layers
            self.convs = nn.ModuleList()
            
            # First layer
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(
                    hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout
                ))
            
            # Last layer (single head)
            if num_layers > 1:
                self.convs.append(GATConv(
                    hidden_dim * heads, hidden_dim, heads=1, dropout=dropout
                ))
            
            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, data):
            """Forward pass."""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GAT layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < self.num_layers - 1:  # No activation on last layer
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # MLP
            x = self.mlp(x)
            
            return x
    
    
    class GINModel(nn.Module):
        """Graph Isomorphism Network for molecular property prediction.
        
        Most expressive GNN architecture (theoretically).
        
        Args:
            input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GIN layers
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.num_layers = num_layers
            self.dropout = dropout
            
            # GIN layers
            self.convs = nn.ModuleList()
            
            for i in range(num_layers):
                if i == 0:
                    mlp = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                else:
                    mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                
                self.convs.append(GINConv(mlp, train_eps=True))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
            
            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, data):
            """Forward pass."""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GIN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # MLP
            x = self.mlp(x)
            
            return x
    
    
    class MPNNModel(nn.Module):
        """Message Passing Neural Network for chemistry.
        
        Based on Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry.
        
        Args:
            input_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of message passing steps
            dropout: Dropout probability
        """
        
        def __init__(
            self,
            input_dim: int = 12,
            edge_dim: int = 4,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Node embedding
            self.node_encoder = nn.Linear(input_dim, hidden_dim)
            
            # Edge embedding
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
            
            # Message passing layers
            self.message_functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                for _ in range(num_layers)
            ])
            
            # Update functions (GRU)
            self.update_functions = nn.ModuleList([
                nn.GRUCell(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            
            # Readout MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, data):
            """Forward pass."""
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            
            # Encode nodes
            h = self.node_encoder(x)
            
            # Message passing
            for mp_layer, update_fn in zip(self.message_functions, self.update_functions):
                # Gather messages
                messages = self._message_pass(h, edge_index, edge_attr, mp_layer)
                
                # Update node states
                h = update_fn(messages, h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Global pooling
            h = global_add_pool(h, batch)
            
            # Readout
            out = self.mlp(h)
            
            return out
        
        def _message_pass(self, h, edge_index, edge_attr, message_fn):
            """Perform message passing step."""
            row, col = edge_index
            
            # Concatenate source, target, and edge features
            messages = torch.cat([h[row], h[col], edge_attr], dim=1)
            
            # Apply message function
            messages = message_fn(messages)
            
            # Aggregate messages
            out = torch.zeros_like(h)
            out.index_add_(0, col, messages)
            
            return out


# Model factory
def create_gnn_model(
    model_type: str = 'gcn',
    **kwargs
) -> 'nn.Module':
    """Create a GNN model.
    
    Args:
        model_type: Type of GNN ('gcn', 'gat', 'gin', 'mpnn')
        **kwargs: Model-specific arguments
    
    Returns:
        GNN model instance
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and PyTorch Geometric required")
    
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GCNModel(**kwargs)
    elif model_type == 'gat':
        return GATModel(**kwargs)
    elif model_type == 'gin':
        return GINModel(**kwargs)
    elif model_type == 'mpnn':
        return MPNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
