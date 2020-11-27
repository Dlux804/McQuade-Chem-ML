import random

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs

from core.storage import cd
from recommender_dev import calculate_fragments, find_similar_molecules, insert_dataset_molecules


def list_similarity(a, b):
    # Make sure lists are being passed
    if not isinstance(a, list) or not isinstance(b, list):
        raise ValueError("Objects passed must be lists")

    # Make sure list are the same length
    if len(a) != len(b):
        raise TypeError("List values are not the same length")

    # Make sure there are no repeating values in lists a and b
    if not len(a) == len(set(a)):
        raise TypeError("There can be no repeating values in given lists")
    if not len(b) == len(set(b)):
        raise TypeError("There can be no repeating values in given lists")

    # Gather values and cast them into dicts storing indexes
    index = []
    for i in range(len(a)):
        index.append(i)
    a = dict(zip(a, index))
    b = dict(zip(b, index))

    # Make sure all values in a are in b and all values in b are in a
    for a_value in a.keys():
        try:
            b[a_value]
        except KeyError:
            raise KeyError(f'Value "{a_value}" not found in list b')
    for b_value in b.keys():
        try:
            a[b_value]
        except KeyError:
            raise KeyError(f'Value "{b_value}" not found in list a')

    # Convert values in list into positional numbers, find where values in list b are relative to list a
    difference_index = []
    for b_val in b.keys():
        difference_index.append(a[b_val])

    # Calculate the max error (different for even and odd length list)
    if (len(a) % 2) == 0:
        n = len(a) / 2
        max_error = 2 * (n ** 2)
    else:
        n = (len(a) - 1) / 2
        max_error = 2 * (n ** 2 + n)

    # Verify calculated max error returned an int
    if int(max_error) == max_error:
        max_error = int(max_error)
    else:
        raise Exception('Bug in calculating max error')

    # Calculate similarity
    a = np.array(index)
    b = np.array(difference_index)
    error = sum(abs(a - b))
    similarity = 1 - error / max_error
    return similarity


# with cd('recommender_test_files/results'):
#
#     runs = []
#     for directory in os.listdir():
#         with cd(directory):
#             control = pd.read_csv('neo4j_pulled_smiles.csv')
#             len_control_frags = len(calculate_fragments(control['smiles'][0]))
#             control_list = control['Unnamed: 0'].to_list()
#             for i in range(5):
#                 test = pd.read_csv(f'neo4j_molecule_{str(i)}.csv')
#                 len_test_frags = len(calculate_fragments(test['smiles'][0]))
#                 test_list = test['Unnamed: 0'].to_list()
#                 sim_score = list_similarity(control_list, test_list)
#                 run_name = f'{directory}_{i}'
#                 runs.append({'run': run_name, 'sim_score': sim_score, 'Number of control fragments': len_control_frags,
#                              'Number of testing fragments': len_test_frags})
#     runs = pd.DataFrame(runs)
#
# runs.to_csv('sim_score_results.csv')


########################################################
data = pd.read_csv('recommender_test_files/lipo_raw.csv')

random.seed(5)
random_int = random.randint(0, len(data))
control_smiles = dict(data.iloc[random_int])['smiles']
data = data.drop(random_int)

neo4j_results = find_similar_molecules(smiles=control_smiles)
neo4j_smiles = neo4j_results['smiles'].to_list()

control_fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(control_smiles))
rdkit_results = []
for smiles in neo4j_smiles:
    testing_fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))
    sim_score = DataStructs.FingerprintSimilarity(control_fingerprint, testing_fingerprint)
    rdkit_results.append({'smiles': smiles, 'sim_score': sim_score})
rdkit_results = pd.DataFrame(rdkit_results)
rdkit_results = rdkit_results.sort_values(by=['sim_score'], ascending=False)
rdkit_smiles = rdkit_results['smiles'].to_list()

rdkit_results.to_csv('rdkit_results.csv')
neo4j_results.to_csv('neo4j_results.csv')
print(list_similarity(rdkit_smiles, neo4j_smiles))
