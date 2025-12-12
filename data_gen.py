import numpy as np
import json

def generate_dat():
    cell_names = np.loadtxt('CCLE_2369_binary_cnv.csv', delimiter=',', usecols=0, skiprows=1, dtype=str)
    dat = np.loadtxt('CCLE_2369_binary_cnv.csv', delimiter=',', usecols=range(1, 2370), skiprows=1) #cell features
    for i in dat:
        i[i == -1] = 0 # turn -1 into 0
    cell_representations = {cell_names[i]: [dat[i], i] for i in range(len(cell_names))}

    drug_representations = {}
    drug_representations_file = 'drug_smiles_tfidf_vectors.json'
    with open(drug_representations_file, 'r') as f:
        drug_info = json.load(f)
    for i in range(1, len(drug_info)):
        drug_representations[drug_info[i]['drug_name']] = [drug_info[i]['tfidf_vector'], i-1]
    drug_features = [drug_representations[i][0] for i in drug_representations.keys()]
    # 0,2,9
    X_data = np.loadtxt('sorted_IC50_82833_580_170.csv', delimiter=',', usecols=(0,2), dtype=str, skiprows=1)
    Y = np.loadtxt('sorted_IC50_82833_580_170.csv', delimiter=',', usecols=9, dtype=float, skiprows=1)

    pair_drug_idx = [drug_representations[X_data[i][1]][1] for i in range(len(X_data))]
    pair_cell_idx = [cell_representations[X_data[i][0]][1] for i in range(len(X_data))]

    return dat, np.asarray(drug_features), np.asarray(pair_cell_idx), np.asarray(pair_drug_idx), Y