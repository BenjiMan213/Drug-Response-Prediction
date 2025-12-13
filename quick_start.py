
import numpy as np
import sys
from pathlib import Path

# Check required files exist
required_files = {
    'cv_pipeline.py': 'CV pipeline module',
    'model_adapter.py': 'Model adapter module',
    'drug_smiles_tfidf_features.csv': 'Drug TF-IDF features'
}
print("\nChecking required files...")

missing_files = []
for file, description in required_files.items():
    if Path(file).exists():
        print(f" {file} ({description})")
    else:
        print(f" {file} ({description}) - MISSING!")
        missing_files.append(file)

if missing_files:
    print(f"\nError: Missing required files: {', '.join(missing_files)}")
    print("Please ensure all files are in the current directory.")
    sys.exit(1)

try:
    from cv_pipeline import (
        load_drug_features_from_csv,
        DrugResponseData,
        run_all_cv
    )

except ImportError as e:
    print(f"Failed to import cv_pipeline: {e}")
    sys.exit(1)

try:
    from model_adapter import SimpleDrugResponseModel, DrugResponseModel, DEFAULT_CONFIG, DeepDrugResponseModel

except ImportError as e:
    print(f"Failed to import model_adapter: {e}")
    sys.exit(1)

import data_gen
cell_features, drug_features, pair_cell_idx, pair_drug_idx, pair_ic50 = data_gen.generate_dat()

n_cells = len(cell_features)
n_drugs = len(drug_features)
n_pairs = len(pair_drug_idx)
cell_feat_dim = len(cell_features[0])  # Example: gene expression features

print(f"\nCell features: {cell_features.shape}")
print(f"Pair data: {n_pairs} cell-drug pairs")
print(f"Cell index range: {0} to {max(pair_cell_idx)}")
print(f"Drug index range: {0} to {max(pair_drug_idx)}")
print(f"IC50 range: {pair_ic50.min():.2f} to {pair_ic50.max():.2f}")

data = DrugResponseData(
    cell_features=cell_features,
    drug_features=drug_features,
    pair_cell_idx=pair_cell_idx,
    pair_drug_idx=pair_drug_idx,
    pair_ic50=pair_ic50
)

# Validate data
data.validate()

# Experiment configuration

config = {
    'model_class': DeepDrugResponseModel,  # or DrugResponseModel if you want
    'split_types': ['random', 'cell_blind', 'drug_blind', 'cell_drug_blind'],
    'n_folds': 5,          # project requirement
    'batch_size': 128,     # can tweak if memory is an issue
    'lr': 1e-3,
    'n_epochs': 50,        # or 50–100 for final
    'device': 'cpu',       # keep 'cpu' on your laptop
    'verbose': True
}

print("\nExperiment configuration:")
for key, value in config.items():
    if key != 'model_class':
        print(f"  {key}: {value}")


results = run_all_cv(**config, data=data)

for split_type, metrics in results.items():
    print(f"\n{split_type.upper()}:")
    print(f"  MSE:  {metrics[:, 0].mean():.4f} ± {metrics[:, 0].std():.4f}")
    print(f"  MAE:  {metrics[:, 1].mean():.4f} ± {metrics[:, 1].std():.4f}")
    print(f"  RMSE: {metrics[:, 2].mean():.4f} ± {metrics[:, 2].std():.4f}")

    print(f"\n  Per-fold results:")
    for fold_idx in range(len(metrics)):
        print(f"    Fold {fold_idx + 1}: MSE={metrics[fold_idx, 0]:.4f}, "
              f"MAE={metrics[fold_idx, 1]:.4f}, RMSE={metrics[fold_idx, 2]:.4f}")


print("Experiment complete!")
