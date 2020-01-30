
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:02:54 2019

@author: quang
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

exp = input('CSV name: ')


def make_fps(csv, smiles_col):
    """
    :param csv:
    :param smiles_col:
    :return:
    """
    df = pd.read_csv(csv)
    df_smiles = df[smiles_col]
    c_smiles = list(map(Chem.CanonSmiles, df_smiles))
    m_l = list(map(Chem.MolFromSmiles, c_smiles))
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in m_l]
    print(fps)
    return fps, c_smiles


def tanimoto_similarity(fps, c_smiles):
    """

    :param fps:
    :param c_smiles:
    :return:
    """
    query, target, similarity = [], [], []
    for n in range(len(fps)-1):
        sim3 = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n+1:])
        print(sim3)
    # sim3 = DataStructs.BulkDiceSimilarity(fps[n], fps[n+1:])
        for m in range(len(sim3)):
            query.append(c_smiles[n])
            target.append(c_smiles[n+1:][m])
            similarity.append(sim3[m])

    return query, target, similarity

def tversky_similarity(fps, c_smiles):
    """

    :param fps:
    :param c_smiles:
    :return:
    """
    query, target, similarity = [], [], []
    for n in range(len(fps)-1):
        sim3 = DataStructs.BulkTverskySimilarity(fps[n], fps[n+1:])
        print(sim3)
    # sim3 = DataStructs.BulkDiceSimilarity(fps[n], fps[n+1:])
        for m in range(len(sim3)):
            query.append(c_smiles[n])
            target.append(c_smiles[n+1:][m])
            similarity.append(sim3[m])

    return query, target, similarity


def return_csv(query, target, similarity):
    """

    :param query:
    :param target:
    :param similarity:
    :return:
    """
    # build the dataframe and sort it
    columns = {'query': query, 'target': target, 'Similarity': similarity}
    df_final = pd.DataFrame(data=columns)
    df_final.to_csv(exp+'.csv', index=False, sep=',')


# if __name__ == "__main__":
#     fps, c_smiles = make_fps('Bulk-Chemicals-Alldata.csv', 'Smiles')
#     query, target, similarity = similarity(fps, c_smiles)
#     return_csv(query, target, similarity)