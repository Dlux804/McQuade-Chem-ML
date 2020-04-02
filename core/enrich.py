"""
This is just a file to featurize the datasets and print them back to CSV.
Just so we have the featurized datasets (using RDKit).

"""

# from core import features
# from core import ingest
from core import models
from core.misc import cd
import os
import csv
import pandas as pd

sets = {
        # 'Lipophilicity-ID.csv': 'exp',
        # 'ESOL.csv': 'water-sol',
        'water-energy.csv': 'expt'
        # 'logP14k.csv': 'Kow',
        # 'jak2_pic50.csv': 'pIC50',
        # 'flashpoint2.csv': 'flashpoint'
    }
alg = 'rf'
method = [0]
for data, target in sets.items():  # loop over dataset dictionary

    # change active directory
    with cd('../dataFiles'):
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)

        # initiate model class with algorithm, dataset and target
        model = models.MlModel(alg, data, target)
        print('done.')

        print('Model Type:', alg)
        print('Featurization:', method)
        print('Dataset:', data)
        print()
        # featurize molecules
        model.featurization(method)
        print(model.data)
        # model.data = model.data.drop(columns=['RDKit2D_calculated'], axis=1)
        # model.data = model.data[['smiles', target, 'ExactMolWt', 'HeavyAtomCount', 'qed']]
        model.data.to_csv('featurized-' + data, index=False)
        node = model.data

    types_dict = node.dtypes.to_dict()
    print(types_dict)
    node['smiles'] = node.smiles.astype(str)
    types_dict = node.dtypes.to_dict()
    cols = list(node.columns)
    print(cols)


    print(types_dict['RDKit2D_calculated'].name, type(types_dict['smiles'].name))
    # print(type(types_dict['smiles'].name))

type_dict = {
    'object': ':STRING',  # strings are stored as objects in numpy
    'float64': ':FLOAT',
    'int64': ':INT',
    'bool': ':BOOLEAN'
}

header = ['smiles:ID']
cols = cols[1:]  # get rid of smiles column header.  Done explicitly
for col in cols:
    head = col + type_dict[types_dict[col].name]
    header.append(head)
print(header)

# headwrt = csv.writer(open('featurized-ESOL-header.csv', 'wb'), delimiter=',')
# headwrt.writerow(header)
# headwrt.close()

with open('featurized-ESOL-header.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)  #, quoting=csv.QUOTE_ALL)
    wr.writerow(header)


