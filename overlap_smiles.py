import pandas as pd
import itertools
from core.misc import cd
import os
from rdkit import Chem
# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

"""
Objective: Find SMILES overlap between multiple files

"""


def overlap_smiles(foldername, datasets):
    """

    :param foldername: Name of the folder which contains all the datasets
    :param datasets: Dictionary of data with their SMILES column
    :return: NONE. CSVs with SMILES overlap between pairs of datasets will be created
    """

    os.chdir(ROOT_DIR)  # Start in root directory
    with cd(foldername):
        all_smiles_dict = {}  # Dictionary for all SMILES list
        for dataset, col in datasets.items():  # Loop over dataset and SMILES header
            print('Working with:', dataset)
            df = pd.read_csv(dataset)
            smiles_list = df[col].tolist()
            mol = list(map(Chem.MolFromSmiles, smiles_list))
            canon_smiles_list = list((map(Chem.MolToSmiles, mol)))
            all_smiles_dict[dataset] = canon_smiles_list  # Make dictionary
        print('Dictionary of all the smiles:', all_smiles_dict)
        all_dict_values = all_smiles_dict.values()  # All SMILES value
        all_dict_keys = list(all_smiles_dict.keys())  # All keys to list
        # Make all keys' list
        key_list = []
        for i in range(2, len(all_dict_keys)+1):
            print("Combining " + str(i) + ' dataset')
            key_combo = list(itertools.combinations(all_dict_keys, i))
            print('All key combinations in ' + str(i) + ' sets', key_combo)
            for key_combination in key_combo:
                iter_combination = list(key_combination)
                # print(iter_combination)
                separator = '_'
                final_str = separator.join(iter_combination)
                print(final_str)
                key_list.append(final_str)
        # Make all combination list
        set_list = []
        for lst_value in all_dict_values:
            set_values = set(lst_value)  # Turn all dict values into set for intersection later
            set_list.append(set_values)
        print(set_list)
        combination_list = []
        for i in range(2, len(set_list)+1):
            print("Combining values for " + str(i) + ' dataset')
            all_value_combo = list(itertools.combinations(set_list, i))
            for value_combo in all_value_combo:
                intersects = list(set.intersection(*list(value_combo)))
                print('Molecules intersected:', len(intersects))
                combination_list.append(intersects)
        for key, combo in zip(key_list, combination_list):
            if len(combo) < 1:
                pass
                # print('There are {0} overlapping molecules for {1}:'.format(len(combo), key))
            else:
                print('There are {0} overlapping molecules for {1}:'.format(len(combo), key))
                overlap_df = pd.DataFrame(combo, columns=[key])
                overlap_df.to_csv("overlap_" + key + '.csv')



data = {
        # "pyridine_cas.csv": "CAS",
        # "pyridine_smi_1.csv": "smiles",
        # "pyridine_smi_2.csv": "smiles",
        "cmc_noadd.csv": "canon_smiles",
        "logP14k.csv": "SMILES",
        "18k-logP.csv": "smiles",
        "ESOL.csv": "smiles",
        "cmc_smiles_26.csv": "smiles",
        "flashpoint.csv": "smiles",
        "Lipophilicity-ID.csv": "smiles",
        "jak2_pic50.csv": "SMILES",
        "water-energy.csv": "smiles"
        # "pyridine_smi_3.csv:"  "smiles"
        }

overlap_smiles('dataFiles', data)
