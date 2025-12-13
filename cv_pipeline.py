
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Callable, Optional
from dataclasses import dataclass
import json
import pickle


@dataclass
class DrugResponseData:
    """
        Container for all drug response data.

        cell_features: Array of shape [n_cells, cell_feat_dim]
        drug_features: Array of shape [n_drugs, drug_feat_dim]
        pair_cell_idx: Array of length N_pairs, values in [0, n_cells-1]
        pair_drug_idx: Array of length N_pairs, values in [0, n_drugs-1]
        pair_ic50: Array of length N_pairs (Log IC50 targets)
    """
    cell_features: np.ndarray
    drug_features: np.ndarray
    pair_cell_idx: np.ndarray
    pair_drug_idx: np.ndarray
    pair_ic50: np.ndarray

    def validate(self):
        """Basic validation of data consistency."""
        n_pairs = len(self.pair_ic50)
        assert len(self.pair_cell_idx) == n_pairs, "Cell idx length mismatch"
        assert len(self.pair_drug_idx) == n_pairs, "Drug idx length mismatch"
        assert self.pair_cell_idx.max() < len(self.cell_features), "Invalid cell idx"
        assert self.pair_drug_idx.max() < len(self.drug_features), "Invalid drug idx"
        print(f"  Data validated: {n_pairs} pairs, {len(self.cell_features)} cells, {len(self.drug_features)} drugs")
        print(f"  Cell features: {self.cell_features.shape}")
        print(f"  Drug features: {self.drug_features.shape}")


class PairDataset(Dataset):
    """Dataset for cell-drug pairs."""

    def __init__(self, pair_indices: np.ndarray, data: DrugResponseData):
        self.pair_indices = pair_indices
        self.data = data

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, i):
        p = self.pair_indices[i]
        c = self.data.pair_cell_idx[p]
        d = self.data.pair_drug_idx[p]

        x_cell = torch.FloatTensor(self.data.cell_features[c])
        x_drug = torch.FloatTensor(self.data.drug_features[d])
        y = torch.FloatTensor([self.data.pair_ic50[p]])

        return x_cell, x_drug, y



def make_random_pair_folds(data: DrugResponseData, n_folds: int = 5,
                           val_fraction: float = 0.1, seed: int = 0) -> List[Tuple]:
    """Standard random 5-fold CV on pairs.

    Returns:
        List of (train_idx, val_idx, test_idx) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)
    n_pairs = len(data.pair_ic50)
    all_idx = np.arange(n_pairs)
    rng.shuffle(all_idx)
    folds = np.array_split(all_idx, n_folds)

    fold_splits = []
    for k in range(n_folds):
        test_idx = folds[k]
        train_all = np.concatenate([folds[i] for i in range(n_folds) if i != k])

        # Carve out validation set
        rng.shuffle(train_all)
        n_val = int(len(train_all) * val_fraction)
        val_idx = train_all[:n_val]
        train_idx = train_all[n_val:]

        fold_splits.append((train_idx, val_idx, test_idx))

    print(f" Random split: {n_folds} folds created")
    return fold_splits


def make_cell_blind_folds(data: DrugResponseData, n_folds: int = 5,
                          val_fraction: float = 0.1, seed: int = 0) -> List[Tuple]:
    """Cell-blind CV: test on unseen cell lines.

    Returns:
        List of (train_idx, val_idx, test_idx) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)
    unique_cells = np.unique(data.pair_cell_idx)
    rng.shuffle(unique_cells)
    cell_folds = np.array_split(unique_cells, n_folds)

    fold_splits = []
    for k in range(n_folds):
        test_cells = set(cell_folds[k])
        test_mask = np.isin(data.pair_cell_idx, list(test_cells))
        test_idx = np.where(test_mask)[0]
        train_idx_all = np.where(~test_mask)[0]

        rng.shuffle(train_idx_all)
        n_val = int(len(train_idx_all) * val_fraction)
        val_idx = train_idx_all[:n_val]
        train_idx = train_idx_all[n_val:]

        fold_splits.append((train_idx, val_idx, test_idx))

    print(f"  Cell-blind split: {n_folds} folds created")
    return fold_splits


def make_drug_blind_folds(data: DrugResponseData, n_folds: int = 5,
                          val_fraction: float = 0.1, seed: int = 0) -> List[Tuple]:
    """Drug-blind CV: test on unseen drugs.

    Returns:
        List of (train_idx, val_idx, test_idx) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)
    unique_drugs = np.unique(data.pair_drug_idx)
    rng.shuffle(unique_drugs)
    drug_folds = np.array_split(unique_drugs, n_folds)

    fold_splits = []
    for k in range(n_folds):
        test_drugs = set(drug_folds[k])
        test_mask = np.isin(data.pair_drug_idx, list(test_drugs))
        test_idx = np.where(test_mask)[0]
        train_idx_all = np.where(~test_mask)[0]

        rng.shuffle(train_idx_all)
        n_val = int(len(train_idx_all) * val_fraction)
        val_idx = train_idx_all[:n_val]
        train_idx = train_idx_all[n_val:]

        fold_splits.append((train_idx, val_idx, test_idx))

    print(f"  Drug-blind split: {n_folds} folds created")
    return fold_splits


def make_cell_drug_blind_folds(data: DrugResponseData, n_folds: int = 5,
                               val_fraction: float = 0.1, seed: int = 0) -> List[Tuple]:
    """Strict cell+drug blind CV: test on pairs with both unseen cell AND unseen drug.

    Train: pairs where cell NOT in test_cells AND drug NOT in test_drugs
    Test: pairs where cell IN test_cells AND drug IN test_drugs

    Returns:
        List of (train_idx, val_idx, test_idx) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)
    unique_cells = np.unique(data.pair_cell_idx)
    unique_drugs = np.unique(data.pair_drug_idx)
    rng.shuffle(unique_cells)
    rng.shuffle(unique_drugs)

    cell_folds = np.array_split(unique_cells, n_folds)
    drug_folds = np.array_split(unique_drugs, n_folds)

    fold_splits = []
    for k in range(n_folds):
        test_cells = set(cell_folds[k])
        test_drugs = set(drug_folds[k])

        # Test: both cell AND drug in test sets
        test_mask = (np.isin(data.pair_cell_idx, list(test_cells)) &
                    np.isin(data.pair_drug_idx, list(test_drugs)))

        # Train: neither cell NOR drug in test sets
        train_mask = (~np.isin(data.pair_cell_idx, list(test_cells)) &
                     ~np.isin(data.pair_drug_idx, list(test_drugs)))

        test_idx = np.where(test_mask)[0]
        train_idx_all = np.where(train_mask)[0]

        rng.shuffle(train_idx_all)
        n_val = int(len(train_idx_all) * val_fraction)
        val_idx = train_idx_all[:n_val]
        train_idx = train_idx_all[n_val:]

        fold_splits.append((train_idx, val_idx, test_idx))

    print(f" Cell+Drug blind split: {n_folds} folds created")
    return fold_splits



def train_one_fold(model_class: type,
                   data: DrugResponseData,
                   train_idx: np.ndarray,
                   val_idx: np.ndarray,
                   test_idx: np.ndarray,
                   batch_size: int = 128,
                   lr: float = 1e-3,
                   n_epochs: int = 50,
                   device: str = "cuda",
                   verbose: bool = True) -> Tuple[float, float, float]:
    """
    Train and evaluate a model on one fold.
    """
    # Create datasets
    train_dataset = PairDataset(train_idx, data)
    val_dataset = PairDataset(val_idx, data)
    test_dataset = PairDataset(test_idx, data)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    cell_dim = data.cell_features.shape[1]
    drug_dim = data.drug_features.shape[1]
    model = model_class(cell_dim, drug_dim).to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_cell, x_drug, y in train_loader:
            x_cell = x_cell.to(device)
            x_drug = x_drug.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x_cell, x_drug)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cell, x_drug, y in val_loader:
                x_cell = x_cell.to(device)
                x_drug = x_drug.to(device)
                y = y.to(device)
                y_pred = model(x_cell, x_drug)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * len(y)

        val_loss /= len(val_dataset)

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    # Load best model for testing
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for x_cell, x_drug, y in test_loader:
            x_cell = x_cell.to(device)
            x_drug = x_drug.to(device)
            y_pred = model(x_cell, x_drug).cpu()
            all_y_true.append(y)
            all_y_pred.append(y_pred)

    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()

    # Compute metrics
    mse = ((y_true - y_pred) ** 2).mean()
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(mse)

    return mse, mae, rmse


def run_cv(config_name: str,
           split_fn: Callable,
           model_class: type,
           data: DrugResponseData,
           n_folds: int = 5,
           **train_kwargs) -> np.ndarray:
    """
    Run complete cross-validation for one configuration.
    """
    print(f"\n{'='*60}")
    print(f"Running {config_name.upper()} Cross-Validation")
    print(f"{'='*60}")

    fold_splits = split_fn(data, n_folds=n_folds)

    all_metrics = []
    for k, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        print(f"\nFold {k+1}/{n_folds}:")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        mse, mae, rmse = train_one_fold(
            model_class, data, train_idx, val_idx, test_idx, **train_kwargs
        )

        all_metrics.append((mse, mae, rmse))
        print(f"  Results: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Summary statistics
    all_metrics = np.array(all_metrics)
    mean = all_metrics.mean(axis=0)
    std = all_metrics.std(axis=0)

    print(f"\n{config_name.upper()} SUMMARY (mean ± std):")
    print(f"  MSE:  {mean[0]:.4f} ± {std[0]:.4f}")
    print(f"  MAE:  {mean[1]:.4f} ± {std[1]:.4f}")
    print(f"  RMSE: {mean[2]:.4f} ± {std[2]:.4f}")

    return all_metrics


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def run_all_cv(model_class: type,
               data: DrugResponseData,
               split_types: List[str] = None,
               n_folds: int = 5,
               **train_kwargs) -> dict:
    """Run all cross-validation configurations.

    Args:
        model_class: PyTorch model class
        data: DrugResponseData object
        split_types: List of split types to run. Options:
                    ["random", "cell_blind", "drug_blind", "cell_drug_blind"]
                    Default: all four
        n_folds: Number of folds (default: 5)
        **train_kwargs: Additional training arguments (batch_size, lr, n_epochs, etc.)

    Returns:
        Dictionary mapping split_type -> metrics array [n_folds, 3]
    """
    if split_types is None:
        split_types = ["random", "cell_blind", "drug_blind", "cell_drug_blind"]

    # Validate data
    data.validate()

    # Map split names to functions
    split_functions = {
        "random": make_random_pair_folds,
        "cell_blind": make_cell_blind_folds,
        "drug_blind": make_drug_blind_folds,
        "cell_drug_blind": make_cell_drug_blind_folds,
    }

    results = {}
    for split_type in split_types:
        if split_type not in split_functions:
            print(f"Warning: Unknown split type '{split_type}', skipping...")
            continue

        metrics = run_cv(
            config_name=split_type,
            split_fn=split_functions[split_type],
            model_class=model_class,
            data=data,
            n_folds=n_folds,
            **train_kwargs
        )
        results[split_type] = metrics

    return results