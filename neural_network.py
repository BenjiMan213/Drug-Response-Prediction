"""
Modular Fully Connected Neural Network for Drug Response Prediction

This module provides a flexible PyTorch-based neural network with customizable:
- Input/output sizes
- Number and size of hidden layers
- Optimizers
- Loss functions
- Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Dict, Callable, Tuple
import numpy as np


class ModularFCNN(nn.Module):
    """
    Fully Connected Neural Network with customizable architecture.

    Args:
        input_size: Number of input features
        hidden_layers: List of integers specifying the size of each hidden layer
        dropout_rate: Dropout probability for regularization (default: 0.0)
        activation: Activation function to use ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        batch_norm: Whether to use batch normalization (default: False)
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.0,
        activation: str = 'relu',
        batch_norm: bool = False
    ):
        super(ModularFCNN, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Define activation function
        self.activation = self._get_activation(activation)

        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None

        # Input layer
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))

            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))

            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer (single value)
        self.output_layer = nn.Linear(prev_size, 1)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }

        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")

        return activations[activation.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if self.batch_norm:
                x = self.batch_norms[i](x)

            x = self.activation(x)

            if self.dropouts:
                x = self.dropouts[i](x)

        x = self.output_layer(x)
        return x

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralNetworkTrainer:
    """
    Trainer class for the Modular FCNN with customizable training components.

    Args:
        model: The neural network model to train
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop', 'adamw')
        loss_fn_name: Name of loss function ('mse', 'mae', 'huber', 'smooth_l1')
        learning_rate: Learning rate for optimizer
        optimizer_params: Additional parameters for optimizer
        device: Device to train on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model: ModularFCNN,
        optimizer_name: str = 'adam',
        loss_fn_name: str = 'mse',
        learning_rate: float = 0.001,
        optimizer_params: Optional[Dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # Set up optimizer
        self.optimizer = self._get_optimizer(optimizer_name, optimizer_params or {})

        # Set up loss function
        self.loss_fn = self._get_loss_function(loss_fn_name)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def _get_optimizer(self, optimizer_name: str, optimizer_params: Dict) -> optim.Optimizer:
        """Get optimizer by name."""
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamw': optim.AdamW,
            'adagrad': optim.Adagrad
        }

        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

        optimizer_class = optimizers[optimizer_name.lower()]
        return optimizer_class(self.model.parameters(), lr=self.learning_rate, **optimizer_params)

    def _get_loss_function(self, loss_fn_name: str) -> nn.Module:
        """Get loss function by name."""
        loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.HuberLoss(),
            'smooth_l1': nn.SmoothL1Loss()
        }

        if loss_fn_name.lower() not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_fn_name}. Choose from {list(loss_functions.keys())}")

        return loss_functions[loss_fn_name.lower()]

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        metrics: Optional[List[Callable]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            metrics: List of metric functions

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Initialize metric accumulators
        metric_values = {f'metric_{i}': 0.0 for i in range(len(metrics or []))}

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.loss_fn(predictions, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Compute metrics
            if metrics:
                predictions_np = predictions.detach().cpu().numpy()
                targets_np = batch_y.detach().cpu().numpy()

                for i, metric_fn in enumerate(metrics):
                    metric_values[f'metric_{i}'] += metric_fn(predictions_np, targets_np)

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metric_values.items()}

        return avg_loss, avg_metrics

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        metrics: Optional[List[Callable]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data
            metrics: List of metric functions

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Initialize metric accumulators
        metric_values = {f'metric_{i}': 0.0 for i in range(len(metrics or []))}

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_x)
                loss = self.loss_fn(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

                # Compute metrics
                if metrics:
                    predictions_np = predictions.detach().cpu().numpy()
                    targets_np = batch_y.detach().cpu().numpy()

                    for i, metric_fn in enumerate(metrics):
                        metric_values[f'metric_{i}'] += metric_fn(predictions_np, targets_np)

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metric_values.items()}

        return avg_loss, avg_metrics

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        metrics: Optional[List[Callable]] = None,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            metrics: List of metric functions
            verbose: Whether to print progress
            early_stopping_patience: Stop if validation loss doesn't improve for N epochs
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, metrics)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)

            # Validate
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader, metrics)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)

                # Early stopping check
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            x: Input tensor

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            predictions = self.model(x)
            return predictions.cpu().numpy()

    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)

    def load_model(self, filepath: str):
        """Load model state dict."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


# Common evaluation metrics
def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_pred - y_true))


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def pearson_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()

    return np.corrcoef(y_pred, y_true)[0, 1]


if __name__ == "__main__":
    # Example usage
    print("Example: Creating a Modular FCNN")
    print("-" * 50)

    # Define network architecture
    input_size = 2048  # e.g., TF-IDF features
    hidden_layers = [512, 256, 128, 64]  # 4 hidden layers

    # Create model
    model = ModularFCNN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=0.3,
        activation='relu',
        batch_norm=True
    )

    print(f"Model created with {model.get_num_parameters():,} parameters")
    print(f"Architecture: {input_size} -> {' -> '.join(map(str, hidden_layers))} -> 1")
    print()

    # Create trainer
    trainer = NeuralNetworkTrainer(
        model=model,
        optimizer_name='adam',
        loss_fn_name='mse',
        learning_rate=0.001,
        optimizer_params={'weight_decay': 1e-5}
    )

    print(f"Trainer initialized with optimizer: Adam, loss: MSE")
    print(f"Device: {trainer.device}")
    print()

    # Create dummy data for demonstration
    dummy_x = torch.randn(100, input_size)
    dummy_y = torch.randn(100, 1)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Define metrics
    metrics = [mean_absolute_error, r_squared]

    print("Training on dummy data...")
    trainer.fit(
        train_loader=train_loader,
        epochs=20,
        metrics=metrics,
        verbose=True
    )

    print()
    print("Training complete!")
    print(f"Final training loss: {trainer.train_losses[-1]:.4f}")
