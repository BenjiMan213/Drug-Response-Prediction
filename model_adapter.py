"""
Drug Response Model Adapter
============================

Adapts the ModularFCNN from neural_network.py to work with the CV pipeline's
dual-input architecture (cell_features + drug_features).

This bridges the gap between:
- neural_network.py: Single-input MLP (designed for drug-only features)
- cv_pipeline.py: Dual-input requirement (cell + drug features)
"""

import torch
import torch.nn as nn
from typing import List, Optional
import sys
import os

# Import the existing neural network module
try:
    from neural_network import ModularFCNN
except ImportError:
    print("Warning: neural_network.py not found. Make sure it's in the same directory.")
    ModularFCNN = None


class DrugResponseModel(nn.Module):
    """
    Dual-input model for drug response prediction.

    This model takes separate cell and drug feature inputs, processes them
    through separate encoders, fuses them, and predicts IC50 values.

    Compatible with cv_pipeline.py's expected interface:
        __init__(cell_feat_dim, drug_feat_dim)
        forward(x_cell, x_drug) -> predictions

    Args:
        cell_feat_dim: Dimension of cell features
        drug_feat_dim: Dimension of drug features (2048 for TF-IDF)
        cell_hidden_layers: Hidden layer sizes for cell encoder
        drug_hidden_layers: Hidden layer sizes for drug encoder
        fusion_hidden_layers: Hidden layer sizes after fusion
        dropout_rate: Dropout probability
        activation: Activation function name
        batch_norm: Whether to use batch normalization
        fusion_method: How to combine cell and drug features ('concat', 'add', 'multiply')
    """

    def __init__(
            self,
            cell_feat_dim: int,
            drug_feat_dim: int,
            cell_hidden_layers: List[int] = [256, 128],
            drug_hidden_layers: List[int] = [512, 256],
            fusion_hidden_layers: List[int] = [256, 128, 64],
            dropout_rate: float = 0.3,
            activation: str = 'relu',
            batch_norm: bool = True,
            fusion_method: str = 'concat'
    ):
        super(DrugResponseModel, self).__init__()

        self.cell_feat_dim = cell_feat_dim
        self.drug_feat_dim = drug_feat_dim
        self.fusion_method = fusion_method

        # Cell encoder
        self.cell_encoder = self._build_encoder(
            cell_feat_dim,
            cell_hidden_layers,
            dropout_rate,
            activation,
            batch_norm
        )

        # Drug encoder
        self.drug_encoder = self._build_encoder(
            drug_feat_dim,
            drug_hidden_layers,
            dropout_rate,
            activation,
            batch_norm
        )

        # Determine fusion input size based on method
        cell_output_dim = cell_hidden_layers[-1] if cell_hidden_layers else cell_feat_dim
        drug_output_dim = drug_hidden_layers[-1] if drug_hidden_layers else drug_feat_dim

        if fusion_method == 'concat':
            fusion_input_dim = cell_output_dim + drug_output_dim
        elif fusion_method in ['add', 'multiply']:
            # For add/multiply, encoders must have same output size
            assert cell_output_dim == drug_output_dim, \
                f"For fusion_method='{fusion_method}', cell and drug encoders must have same output dim"
            fusion_input_dim = cell_output_dim
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        # Fusion network (predictor)
        self.fusion_network = self._build_encoder(
            fusion_input_dim,
            fusion_hidden_layers + [1],  # Add output layer
            dropout_rate,
            activation,
            batch_norm,
            is_predictor=True
        )

    def _build_encoder(
            self,
            input_dim: int,
            hidden_layers: List[int],
            dropout_rate: float,
            activation: str,
            batch_norm: bool,
            is_predictor: bool = False
    ) -> nn.Sequential:
        """Build an encoder network."""
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Don't add activation/batch_norm/dropout after final layer of predictor
            if not (is_predictor and i == len(hidden_layers) - 1):
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                layers.append(self._get_activation(activation))

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        return nn.Sequential(*layers)

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
        return activations.get(activation.lower(), nn.ReLU())

    def forward(self, x_cell: torch.Tensor, x_drug: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_cell: Cell features [batch_size, cell_feat_dim]
            x_drug: Drug features [batch_size, drug_feat_dim]

        Returns:
            Predictions [batch_size, 1]
        """
        # Encode cell and drug features
        cell_embedding = self.cell_encoder(x_cell)
        drug_embedding = self.drug_encoder(x_drug)

        # Fuse embeddings
        if self.fusion_method == 'concat':
            fused = torch.cat([cell_embedding, drug_embedding], dim=1)
        elif self.fusion_method == 'add':
            fused = cell_embedding + drug_embedding
        elif self.fusion_method == 'multiply':
            fused = cell_embedding * drug_embedding
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        # Predict
        output = self.fusion_network(fused)
        return output

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleDrugResponseModel(nn.Module):
    """
    Simplified dual-input model with minimal architecture.
    Good baseline for quick experiments.

    Args:
        cell_feat_dim: Dimension of cell features
        drug_feat_dim: Dimension of drug features (2048 for TF-IDF)
    """

    def __init__(self, cell_feat_dim: int, drug_feat_dim: int):
        super(SimpleDrugResponseModel, self).__init__()

        # Simple encoders
        self.cell_fc = nn.Linear(cell_feat_dim, 128)
        self.drug_fc = nn.Linear(drug_feat_dim, 128)

        # Fusion and prediction
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cell: torch.Tensor, x_drug: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        cell_emb = torch.relu(self.cell_fc(x_cell))
        drug_emb = torch.relu(self.drug_fc(x_drug))
        combined = torch.cat([cell_emb, drug_emb], dim=1)
        return self.predictor(combined)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepDrugResponseModel(nn.Module):
    """
    Deep dual-input model with larger capacity.
    For more complex datasets and when you have sufficient data.

    Args:
        cell_feat_dim: Dimension of cell features
        drug_feat_dim: Dimension of drug features (2048 for TF-IDF)
    """

    def __init__(self, cell_feat_dim: int, drug_feat_dim: int):
        super(DeepDrugResponseModel, self).__init__()

        # Deep cell encoder
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Deep drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fusion and prediction
        self.predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cell: torch.Tensor, x_drug: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        cell_emb = self.cell_encoder(x_cell)
        drug_emb = self.drug_encoder(x_drug)
        combined = torch.cat([cell_emb, drug_emb], dim=1)
        return self.predictor(combined)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example configurations for the DrugResponseModel
DEFAULT_CONFIG = {
    'cell_hidden_layers': [256, 128],
    'drug_hidden_layers': [512, 256],
    'fusion_hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'activation': 'relu',
    'batch_norm': True,
    'fusion_method': 'concat'
}

LIGHTWEIGHT_CONFIG = {
    'cell_hidden_layers': [128],
    'drug_hidden_layers': [256, 128],
    'fusion_hidden_layers': [128, 64],
    'dropout_rate': 0.2,
    'activation': 'relu',
    'batch_norm': False,
    'fusion_method': 'concat'
}

DEEP_CONFIG = {
    'cell_hidden_layers': [512, 256, 128],
    'drug_hidden_layers': [1024, 512, 256],
    'fusion_hidden_layers': [512, 256, 128, 64],
    'dropout_rate': 0.4,
    'activation': 'relu',
    'batch_norm': True,
    'fusion_method': 'concat'
}

if __name__ == "__main__":
    print("=" * 60)
    print("Drug Response Model Adapter - Example Usage")
    print("=" * 60)

    # Example dimensions
    cell_feat_dim = 978  # e.g., gene expression features
    drug_feat_dim = 2048  # TF-IDF features
    batch_size = 32

    print(f"\nInput dimensions:")
    print(f"  Cell features: {cell_feat_dim}")
    print(f"  Drug features: {drug_feat_dim}")
    print(f"  Batch size: {batch_size}")

    # Test all three models
    models = {
        'Simple': SimpleDrugResponseModel(cell_feat_dim, drug_feat_dim),
        'Default': DrugResponseModel(cell_feat_dim, drug_feat_dim, **DEFAULT_CONFIG),
        'Deep': DeepDrugResponseModel(cell_feat_dim, drug_feat_dim)
    }

    print(f"\n{'=' * 60}")
    print("Model Comparison")
    print(f"{'=' * 60}")

    for name, model in models.items():
        print(f"\n{name} Model:")
        print(f"  Parameters: {model.get_num_parameters():,}")

        # Test forward pass
        x_cell = torch.randn(batch_size, cell_feat_dim)
        x_drug = torch.randn(batch_size, drug_feat_dim)

        output = model(x_cell, x_drug)
        print(f"  Output shape: {output.shape}")
        print(f"  Sample predictions: {output[:3, 0].tolist()}")

    print(f"\n{'=' * 60}")
    print("Integration with CV Pipeline")
    print(f"{'=' * 60}")

    print("""
To use with cv_pipeline.py:

from model_adapter import DrugResponseModel, DEFAULT_CONFIG
from cv_pipeline import load_drug_features_from_csv, DrugResponseData, run_all_cv

# Load data
drug_features, drug_name_to_idx, cid_to_idx = load_drug_features_from_csv(
    "drug_smiles_tfidf_features.csv"
)
cell_features = np.load("cell_features.npy")
# ... load pair data ...

# Wrap data
data = DrugResponseData(
    cell_features=cell_features,
    drug_features=drug_features,
    pair_cell_idx=pair_cell_idx,
    pair_drug_idx=pair_drug_idx,
    pair_ic50=pair_ic50
)

# Run experiments with your choice of model
results = run_all_cv(
    model_class=DrugResponseModel,  # or SimpleDrugResponseModel, DeepDrugResponseModel
    data=data,
    n_folds=5,
    batch_size=128,
    lr=1e-3,
    n_epochs=50,
    device="cuda"
)
""")

    print("\n" + "=" * 60)
    print("Available Model Classes:")
    print("=" * 60)
    print("1. SimpleDrugResponseModel - Fast baseline (~200K params)")
    print("2. DrugResponseModel - Flexible, configurable (~500K-2M params)")
    print("3. DeepDrugResponseModel - High capacity (~3M params)")