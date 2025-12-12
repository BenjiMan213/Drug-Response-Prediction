"""
Quick Start Script for Drug Response Prediction
================================================

This script runs a complete experiment from data loading to results.
Modify the file paths and hyperparameters as needed.

Usage:
    python quick_start.py
"""

import numpy as np
import sys
from pathlib import Path

# Check required files exist
required_files = {
    'cv_pipeline.py': 'CV pipeline module',
    'model_adapter.py': 'Model adapter module',
    'drug_smiles_tfidf_features.csv': 'Drug TF-IDF features'
}

print("=" * 60)
print("Drug Response Prediction - Quick Start")
print("=" * 60)
print("\nChecking required files...")

missing_files = []
for file, description in required_files.items():
    if Path(file).exists():
        print(f"✓ {file} ({description})")
    else:
        print(f"✗ {file} ({description}) - MISSING!")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Error: Missing required files: {', '.join(missing_files)}")
    print("Please ensure all files are in the current directory.")
    sys.exit(1)

print("\n" + "=" * 60)
print("Loading modules...")
print("=" * 60)

try:
    from cv_pipeline import (
        load_drug_features_from_csv,
        DrugResponseData,
        run_all_cv
    )

    print("✓ CV pipeline loaded")
except ImportError as e:
    print(f"✗ Failed to import cv_pipeline: {e}")
    sys.exit(1)

try:
    from model_adapter import SimpleDrugResponseModel, DrugResponseModel, DEFAULT_CONFIG

    print("✓ Model adapter loaded")
except ImportError as e:
    print(f"✗ Failed to import model_adapter: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 1: Loading Drug Features")
print("=" * 60)

# Load drug features
#drug_features, drug_name_to_idx, cid_to_idx = load_drug_features_from_csv(
#"drug_smiles_tfidf_features.csv"
#)



print("\n" + "=" * 60)
print("Step 2: Loading Cell Features and Pair Data")
print("=" * 60)

# TODO: Replace with your actual data loading
# For now, create mock data for demonstration
print("\n⚠️  Using MOCK DATA for demonstration")
print("   Replace this section with your actual data loading!")



# Create mock data (REPLACE THIS!)
'''cell_features = np.random.randn(n_cells, cell_feat_dim).astype(np.float32)
pair_cell_idx = np.random.randint(0, n_cells, n_pairs)
pair_drug_idx = np.random.randint(0, n_drugs, n_pairs)
pair_ic50 = np.random.randn(n_pairs).astype(np.float32)'''

import data_gen
cell_features, drug_features, pair_cell_idx, pair_drug_idx, pair_ic50 = data_gen.generate_dat()

n_cells = len(cell_features)
n_drugs = len(drug_features)
n_pairs = len(pair_drug_idx)
cell_feat_dim = len(cell_features[0])  # Example: gene expression features

print(f"\n✓ Cell features: {cell_features.shape}")
print(f"✓ Pair data: {n_pairs} cell-drug pairs")
print(f"  Cell index range: {0} to {max(pair_cell_idx)}")
print(f"  Drug index range: {0} to {max(pair_drug_idx)}")
print(f"  IC50 range: {pair_ic50.min():.2f} to {pair_ic50.max():.2f}")

print("\n" + "=" * 60)
print("Step 3: Creating Data Object")
print("=" * 60)

data = DrugResponseData(
    cell_features=cell_features,
    drug_features=drug_features,
    pair_cell_idx=pair_cell_idx,
    pair_drug_idx=pair_drug_idx,
    pair_ic50=pair_ic50
)

# Validate data
data.validate()

print("\n" + "=" * 60)
print("Step 4: Configuring Experiment")
print("=" * 60)

# Experiment configuration
config = {
    'model_class': SimpleDrugResponseModel,
    'split_types': ['random', 'cell_blind'],  # Start with 2 splits for speed
    'n_folds': 3,  # Use 3 folds for quick test (change to 5 for final)
    'batch_size': 64,
    'lr': 1e-3,
    'n_epochs': 10,  # Use 10 epochs for quick test (change to 50+ for final)
    'device': 'cpu',  # Change to 'cpu' if no GPU
    'verbose': True
}

print("\nExperiment configuration:")
for key, value in config.items():
    if key != 'model_class':
        print(f"  {key}: {value}")
print(f"  model_class: SimpleDrugResponseModel")

print("\n⚠️  NOTE: Using 3 folds × 10 epochs for quick testing")
print("   For final results, use 5 folds × 50+ epochs")

print("\n" + "=" * 60)
print("Step 5: Running Cross-Validation")
print("=" * 60)
print("\nThis may take a few minutes...")

results = run_all_cv(**config, data=data)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for split_type, metrics in results.items():
    print(f"\n{split_type.upper()}:")
    print(f"  MSE:  {metrics[:, 0].mean():.4f} ± {metrics[:, 0].std():.4f}")
    print(f"  MAE:  {metrics[:, 1].mean():.4f} ± {metrics[:, 1].std():.4f}")
    print(f"  RMSE: {metrics[:, 2].mean():.4f} ± {metrics[:, 2].std():.4f}")

    print(f"\n  Per-fold results:")
    for fold_idx in range(len(metrics)):
        print(f"    Fold {fold_idx + 1}: MSE={metrics[fold_idx, 0]:.4f}, "
              f"MAE={metrics[fold_idx, 1]:.4f}, RMSE={metrics[fold_idx, 2]:.4f}")

print("\n" + "=" * 60)
print("Next Steps")
print("=" * 60)
print("""
1. Replace mock data with your actual:
   - cell_features from cell encoding team
   - pair_cell_idx, pair_drug_idx, pair_ic50 from data team

2. Tune hyperparameters:
   - Increase n_folds to 5
   - Increase n_epochs to 50-100
   - Try different learning rates (1e-4, 1e-3, 1e-2)
   - Experiment with batch_size (32, 64, 128, 256)

3. Try different models:
   from model_adapter import DrugResponseModel, DeepDrugResponseModel

4. Run all split types:
   split_types = ['random', 'cell_blind', 'drug_blind', 'cell_drug_blind']

5. Save results:
   import pickle
   with open('results.pkl', 'wb') as f:
       pickle.dump(results, f)
""")

print("\n✓ Experiment complete!")
