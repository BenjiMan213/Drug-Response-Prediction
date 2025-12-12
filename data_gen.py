import numpy as np
import json

cell_names = np.loadtxt('CCLE_2369_binary_cnv.csv', delimiter=',', usecols=0, skiprows=1, dtype=str)
dat = np.loadtxt('CCLE_2369_binary_cnv.csv', delimiter=',', usecols=range(1, 2370), skiprows=1)
for i in dat:
    i[i == -1] = 0 # turn -1 into 0
cell_representations = {cell_names[i]: dat[i] for i in range(len(cell_names))}

drug_representations = {}
drug_representations_file = 'drug_smiles_tfidf_vectors.json'
with open(drug_representations_file, 'r') as f:
    drug_info = json.load(f)
for i in range(1,len(drug_info)):
    drug_representations[drug_info[i]['drug_name']] = drug_info[i]['tfidf_vector']

# 0,2,9
X_data = np.loadtxt('sorted_IC50_82833_580_170.csv', delimiter=',', usecols=(0,2), dtype=str, skiprows=1)
Y = np.loadtxt('sorted_IC50_82833_580_170.csv', delimiter=',', usecols=9, dtype=float, skiprows=1)
cell_features = [cell_representations[X_data[i][0]] for i in range(len(X_data))]
drug_features = [drug_representations[X_data[i][1]] for i in range(len(X_data))]