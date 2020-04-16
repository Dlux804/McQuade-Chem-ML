import pandas as pd
import itertools
from core.misc import cd
import os
from rdkit import Chem
# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

"""
Objective: Make a csv of all the molecules

"""


def all_data(foldername, datasets):
    """

    :param foldername: Name of the folder which contains all the datasets
    :param datasets: Dictionary of data with their SMILES column
    :return: NONE. CSVs with SMILES overlap between pairs of datasets will be created
    """

    os.chdir(ROOT_DIR)  # Start in root directory
    with cd(foldername):
        combined_list = []  # Dictionary for all SMILES list
        for dataset, col in datasets.items():  # Loop over dataset and SMILES header
            print('Working with:', dataset)
            df = pd.read_csv(dataset)
            smiles_list = df[col].tolist()
            mol = list(map(Chem.MolFromSmiles, smiles_list))
            canon_smiles_list = sorted(set(map(Chem.MolToSmiles, mol)))
            combined_list += canon_smiles_list
        final_smiles = set(combined_list)
        print('There are {0} molecules across all dataset'.format(len(final_smiles)))
        all_smiles_df = pd.DataFrame(list(final_smiles), columns=['smiles'])
        all_smiles_df.to_csv('all_smiles.csv')

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

all_data('dataFiles', data)