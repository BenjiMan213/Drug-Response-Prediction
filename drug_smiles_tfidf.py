"""
Apply TF-IDF to chemically normalized isomeric drug SMILES using chemical-word tokenization.

Input CSV format: drug_name, CID, CanonicalSMILES, IsomericSMILES

Outputs:
1) CSV: drug_name, CID, (up to x n-gram TF-IDF features)
2) JSON: list of {"drug_name": ..., "CID": ..., "vector": [x floats]}
3) PKL: pickled fitted TfidfVectorizer
"""

import argparse
import json
import pickle
import re

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.feature_extraction.text import TfidfVectorizer

SMILES_TOKEN_PATTERN = re.compile(
    r'\[[^\]]+\]'       # bracket expressions, e.g. [nH], [O-], [C@H]
    r'|Br|Cl|Si|Se'     # two-letter atoms
    r'|@@?|==?|##?'     # stereo and multiple-bond markers
    r'|[%][0-9]{2}'     # two-digit ring numbers, e.g. %10
    r'|[0-9]'           # ring digits
    r'|[=\\/#\-]'       # bond symbols
    r'|\(|\)'           # branches
    r'|[A-Za-z]'        # single-letter atoms
)
uncharger = rdMolStandardize.Uncharger()
te = rdMolStandardize.TautomerEnumerator()


def standardize_smiles(smiles):
    """
    Standardize a SMILES string:
    1. Parse to RDKit Mol
    2. Cleanup (normalize, reionize)
    3. Keep parent fragment (strip salts/counterions)
    4. Uncharge
    5. Canonical tautomer
    6. Export canonical isomeric SMILES

    Returns:
        standardized canonical isomeric SMILES, or None if parsing fails.
    """
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Cleanup (functional group normalization, reionization, etc.)
    clean_mol = rdMolStandardize.Cleanup(mol)

    # Keep parent (strip salts / inorganic counterions)
    parent = rdMolStandardize.FragmentParent(clean_mol)

    # Uncharge
    uncharged = uncharger.uncharge(parent)

    # Canonical tautomer
    tautomer = te.Canonicalize(uncharged)

    # Canonical isomeric SMILES
    std_smiles = Chem.MolToSmiles(
        tautomer,
        canonical=True,
        isomericSmiles=True
    )
    return std_smiles


def re_tokenize(smiles):
    return SMILES_TOKEN_PATTERN.findall(smiles)


def build_tfidf_features(smiles_list, ngram_range, max_features):
    """
    Fit a TF-IDF vectorizer on a list of SMILES strings using chemical-word tokenization.

    Returns:
        X is a scipy sparse matrix of shape (n_samples, n_features), fitted TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=re_tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram_range,
        max_features=max_features
    )
    X = vectorizer.fit_transform(smiles_list)
    return X, vectorizer


def main():
    parser = argparse.ArgumentParser(
        description="Apply TF-IDF to normalized isomeric drug SMILES using DeepChem's SmilesTokenizer."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV with columns: drug_name, CID, CanonicalSMILES, IsomericSMILES"
    )
    parser.add_argument(
        "start_n_gram",
        default=2,
        help="Input integer for the start of the n-gram range parameter of TF-IDF"
    )
    parser.add_argument(
        "end_n_gram",
        default=4,
        help="Input integer for the end of the n-gram range parameter of TF-IDF"
    )
    parser.add_argument(
        "max_features",
        default=2048,
        help="Input integer for the max features parameter of TF-IDF"
    )
    parser.add_argument(
        "--csv_out",
        default="drug_smiles_tfidf_features.csv",
        help="Output CSV file for TF-IDF features (default: drug_smiles_tfidf_features.csv)"
    )
    parser.add_argument(
        "--json_out",
        default="drug_smiles_tfidf_vectors.json",
        help="Output JSON file for TF-IDF vectors (default: drug_smiles_tfidf_vectors.json)"
    )
    parser.add_argument(
        "--pkl_out",
        default="drug_smiles_tfidf_vectorizer.pkl",
        help="Output pickle file for the fitted TF-IDF vectorizer (default: drug_smiles_tfidf_vectorizer.pkl)"
    )
    args = parser.parse_args()

    ngram_range = (int(args.start_n_gram), int(args.end_n_gram))
    max_features = int(args.max_features)

    # Load input CSV
    df = pd.read_csv(args.input_csv)

    drug_names = []
    cids = []
    standardized_smiles = []

    # Normalize SMILES, preferring IsomericSMILES when available
    for idx, row in df.iterrows():
        iso = row["IsomericSMILES"]
        can = row["CanonicalSMILES"]
        raw_smiles = iso if isinstance(iso, str) and iso.strip() != "" else can

        std = standardize_smiles(raw_smiles)
        if std is None:
            # Skip molecules we can't parse/standardize
            print(f"[WARN] Could not standardize SMILES for row {idx}, CID={row['CID']}. Skipping.")
            continue

        drug_names.append(row["drug_name"])
        cids.append(row["CID"])
        standardized_smiles.append(std)

    if not standardized_smiles:
        raise RuntimeError("No valid SMILES after standardization; nothing to vectorize.")

    # TF-IDF on standardized isomeric SMILES
    X, vectorizer = build_tfidf_features(
        standardized_smiles,
        ngram_range=ngram_range,
        max_features=max_features
    )

    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"Number of features (n-grams): {len(feature_names)}")

    X_dense = X.toarray()
    feats_df = pd.DataFrame(X_dense, columns=feature_names)
    feats_df.insert(0, "StandardizedSMILES", standardized_smiles)
    feats_df.insert(0, "CID", cids)
    feats_df.insert(0, "drug_name", drug_names)
    feats_df.to_csv(args.csv_out, index=False)
    print(f"Saved CSV features to: {args.csv_out}")

    json_records = [{"feature_names": feature_names.tolist()}]
    for i in range(X_dense.shape[0]):
        json_records.append({
            "drug_name": str(drug_names[i]),
            "CID": str(cids[i]),
            "StandardizedSMILES": str(standardized_smiles[i]),
            "tfidf_vector": X_dense[i].tolist()
        })

    def repl_func(match):
        return " ".join(match.group().split())

    with open(args.json_out, 'w', encoding='utf8') as f:
        content = json.dumps(json_records, indent=2)
        content = re.sub(r"(?<=\[)[^\[\]]+(?=\])", repl_func, content)
        f.write(content)
    print(f"Saved JSON vectors to: {args.json_out}")

    with open(args.pkl_out, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF vectorizer pickle to: {args.pkl_out}")


if __name__ == "__main__":
    main()
