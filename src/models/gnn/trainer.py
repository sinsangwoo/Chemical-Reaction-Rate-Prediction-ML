"""Training pipeline for GNN models."""

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch_geometric.loader import DataLoader
    import numpy as np
    from tqdm import tqdm
    from typing import Dict, List, Optional
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class GNNTrainer:
        """Trainer for GNN models.
        
        Args:
            model: GNN model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        
        def __init__(
            self,
            model: nn.Module,
            device: str = 'cpu',
            learning_rate: float = 0.001,
            weight_decay: float = 1e-5
        ):
            self.model = model.to(device)
            self.device = device
            
            # Optimizer
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Learning rate scheduler
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            
            # Loss function
            self.criterion = nn.MSELoss()
            
            # Training history
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_mae': [],
                'val_mae': []
            }
        
        def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
            """Train for one epoch.
            
            Args:
                train_loader: Training data loader
            
            Returns:
                Dictionary of training metrics
            """
            self.model.train()
            total_loss = 0
            total_mae = 0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch).squeeze()
                
                # Compute loss
                loss = self.criterion(predictions, batch.y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(predictions - batch.y)).item()
                num_batches += 1
            
            return {
                'loss': total_loss / num_batches,
                'mae': total_mae / num_batches
            }
        
        def validate(self, val_loader: DataLoader) -> Dict[str, float]:
            """Validate the model.
            
            Args:
                val_loader: Validation data loader
            
            Returns:
                Dictionary of validation metrics
            """
            self.model.eval()
            total_loss = 0
            total_mae = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    predictions = self.model(batch).squeeze()
                    
                    # Compute loss
                    loss = self.criterion(predictions, batch.y)
                    
                    # Metrics
                    total_loss += loss.item()
                    total_mae += torch.mean(torch.abs(predictions - batch.y)).item()
                    num_batches += 1
            
            return {
                'loss': total_loss / num_batches,
                'mae': total_mae / num_batches
            }
        
        def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 100,
            early_stopping_patience: int = 20
        ) -> Dict[str, List[float]]:
            """Train the model.
            
            Args:
                train_loader: Training data loader
                val_loader: Validation data loader
                num_epochs: Number of epochs
                early_stopping_patience: Patience for early stopping
            
            Returns:
                Training history
            """
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Update scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Save history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['train_mae'].append(train_metrics['mae'])
                self.history['val_mae'].append(val_metrics['mae'])
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}:")
                    print(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
                    print(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}")
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            return self.history
        
        def predict(self, test_loader: DataLoader) -> np.ndarray:
            """Make predictions.
            
            Args:
                test_loader: Test data loader
            
            Returns:
                Predictions array
            """
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    pred = self.model(batch).squeeze()
                    predictions.append(pred.cpu().numpy())
            
            return np.concatenate(predictions)
        
        def save_checkpoint(self, filepath: str):
            """Save model checkpoint."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': self.history
            }, filepath)
        
        def load_checkpoint(self, filepath: str):
            """Load model checkpoint."""
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
