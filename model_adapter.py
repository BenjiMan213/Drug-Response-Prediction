
import torch
import torch.nn as nn
from typing import List, Optional
import sys
import os

class DeepDrugResponseModel(nn.Module):

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
            nn.Linear(drug_feat_dim, 512),
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
'''


DEEP_CONFIG = {
    'cell_hidden_layers': [512, 256, 128],
    'drug_hidden_layers': [1024, 512, 256],
    'fusion_hidden_layers': [512, 256, 128, 64],
    'dropout_rate': 0.4,
    'activation': 'relu',
    'batch_norm': True,
    'fusion_method': 'concat'
}'''