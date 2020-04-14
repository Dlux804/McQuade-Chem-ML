import pandas as pd
import itertools
from core.misc import cd
import os

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
            all_smiles_dict[dataset] = smiles_list  # Make dictionary
        print('Dictionary of all the smiles:', all_smiles_dict)
        all_dict_values = list(all_smiles_dict.values())  # All SMILES value
        all_dict_keys = list(all_smiles_dict.keys())  # All keys
        # print(all_lists)
        intersect_list = []  # List of lists od SMILES interseciton
        for a, b in itertools.combinations(all_dict_values, 2):  # Compare datasets in pairs
            print('Comparing:')
            print(all_dict_keys[all_dict_values.index(a)])   # Key for the one of the two dataset used for comparison
            print('and')
            print(all_dict_keys[all_dict_values.index(b)])  # Key for the other dataset used for comparison
            intersects = list(set(a).intersection(b))  # list of SMILES intersection
            # print(intersects)
            intersect_list.append(intersects)
            print('')
        print(intersect_list)
        combo_list = []
        for a, b in itertools.combinations(all_dict_keys, 2):  # Loop over pairs of key to make column name for csvs
            combos = a + '_' + b  # NameFirstDataset_NameSecondDataset
            combo_list.append(combos)
        for intersect, combo in zip(intersect_list, combo_list):
            if len(intersect) < 1:
                pass  # Ignore empty list
            else:
                overlap_df = pd.DataFrame(intersect, columns=[combo])
                overlap_df.to_csv("overlap_" + combo + '.csv')



data = {
        # "pyridine_cas.csv": "CAS",
        # "pyridine_smi_1.csv": "smiles",
        # "pyridine_smi_2.csv": "smiles",
        # "cmc_noadd.csv": "canon_smiles",
        "logP14k.csv": "SMILES",
        # "18k-logP.csv": "smiles",x
        "ESOL.csv": "smiles",
        # "cmc_smiles_26.csv": "smiles",
        # "flashpoint.csv": "smiles",
        "Lipophilicity-ID.csv": "smiles",
        "jak2_pic50.csv": "SMILES",
        "water-energy.csv": "smiles"
        # "pyridine_smi_3.csv" : "smiles"
        }

# overlap_smiles('dataFiles', data)