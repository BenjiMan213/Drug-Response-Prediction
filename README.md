# Drug-Response-Prediction

Dataset download link: https://creammist.mtms.dev/doc/bulk/

## drug_smiles_tfidf.py
Apply TF-IDF to chemically normalized isomeric drug SMILES using chemical-word tokenization.

Input CSV format: drug_name, CID, CanonicalSMILES, IsomericSMILES

Outputs:
1) CSV : drug_name, CID, (up to x n-gram TF-IDF features)
2) JSON : list of {"drug_name": ..., "CID": ..., "vector": [x floats]}
3) PKL : pickled fitted TfidfVectorizer

Notes:
1) Isomeric over only canonical smiles: https://pubs.acs.org/doi/10.1021/acs.jcim.4c00318
2) Standardize smiles/molecules by parent structure (salt stripping), neutralization (remove charge), and tautomer canonicalization: https://link.springer.com/article/10.1186/s13321-022-00606-7
3) Chemical "words" over character n-grams: https://onlinelibrary.wiley.com/doi/10.1002/minf.202300249
4) Word n-grams range of 2 to 4: https://ijai.iaescore.com/index.php/IJAI/article/view/25071