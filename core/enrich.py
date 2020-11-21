# """
# This is just a file to featurize the datasets and print them back to CSV.
# Just so we have the featurized datasets (using RDKit).
#
# """
#
# # from core import features
# # from core import ingest
# from core import models
# from core.storage.misc import cd
# import os
#
# sets = {
#         'lipo_raw.csv': 'exp',
#         'ESOL.csv': 'water-sol',
#         'water-energy.csv': 'expt',
#         'logP14k.csv': 'Kow',
#         'jak2_pic50.csv': 'pIC50',
#         'flashpoint2.csv': 'flashpoint'
#     }
# alg = 'rf'
# method = [0]
# for data, target in sets.items():  # loop over dataset dictionary
#
#     # change active directory
#     with cd('../dataFiles'):
#         print('Now in:', os.getcwd())
#         print('Initializing model...', end=' ', flush=True)
#
#         # initiate model class with algorithm, dataset and target
#         model = models.MlModel(alg, data, target)
#         print('done.')
#
#         print('Model Type:', alg)
#         print('Featurization:', method)
#         print('Dataset:', data)
#         print()
#         # featurize molecules
#         model.featurization(method)
#         print(model.data)
#         model.data = model.data.drop(columns=['RDKit2D_calculated'], axis=1)
#         # model.data = model.data[['smiles', target, 'ExactMolWt', 'HeavyAtomCount', 'qed']]
#         model.data.to_csv('featurized-' + data, index=False)


