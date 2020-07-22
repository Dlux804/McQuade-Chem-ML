# TODO: Make main function that asks user what models they would like to initiate

from core import models, Get_Classification
from core.misc import cd
from core.storage import pickle_model, unpickle_model
import os
import pathlib
from time import sleep
from core.features import featurize

import pandas as pd

# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # Asking the user to decide between running classification models or regression models
    c = str(input("Enter c for classification, r for regression: "))

    # Sets up learner and datasets for classification.
    if c == 'c':
        # list of available classification learning algorithms
        learner = ['svc', 'knc', 'rfc']
        learner = ['knc'] # Use this line to test specific models instead of iterating


        sets = {
#            'BBBP.csv': 'p_np',
            'sider.csv': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
                   'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders',
                   'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'General disorders and administration site conditions', 'Endocrine disorders',
                   'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
                   'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders',
                   'Cardiac disorders',
                   'Nervous system disorders', 'Injury, poisoning and procedural complications'],
#            'clintox.csv': ['FDA_APPROVED', 'CT_TOX'],
 #           'bace.csv': 'Class',
        }


    # Sets up learner, featurizations, and data sets for regression
    if c == 'r':
        # list of available regression learning algorithms
        learner = ['ada', 'rf', 'svr', 'gdb', 'nn', 'knn']
        #learner = ['gdb', 'nn']


        # list of available featurization methods
        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4],
                 [5]]
        feats = [[0]]#, [0, 2]]  # Change this to change which featurizations are being tested (for regression)

        # regression data sets in dict. Key: Filename.csv , Value: Target column header
        sets = {
            'ESOL.csv': 'water-sol',
            # 'Lipophilicity-ID.csv': 'exp',
            # 'water-energy.csv': 'expt',
            # 'logP14k.csv': 'Kow',
            # 'jak2_pic50.csv': 'pIC50'
        }

    for alg in learner:  # loop over all learning algorithms
        # The following if statements set featurization options based on if the
        # model needs normalized data (currently only set up for the classification models)
        if c == 'c':
            feats = Get_Classification.get_classification_feats(alg) # Selects featurizations for classification based on the model being ran

        for method in feats:  # loop over the featurization methods
            for data, target in sets.items(): # loop over dataset dictionary

                if (data == 'sider.csv' or data == 'clintox.csv') and alg == 'svc':
                    pass
                else:
                    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                        print('Model Type:', alg)
                        print('Featurization:', method)
                        print('Dataset:', data)
                        print('Target(s):', target)
                        print()
                        print('Initializing model...', end=' ', flush=True)
                        # initiate model class with algorithm, dataset and target
                        opt_iter = 10
                        if data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv',
                                    'jak2_pic50.csv']:
                            opt_iter = 25
                        model = models.MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                                               tune=False, cv=3, opt_iter=opt_iter)
                        print('Done.\n')

                    with cd('dataFiles'):
                        # Runs classification model
                        model.featurize()  # Featurize molecules
                        val = 0.0
                        if alg == 'nn':
                            val = 0.1
                        model.data_split(val=val)
                        model.reg()
                        model.run()  # Runs the models/featurizations for classification
                        model.analyze()
                        if model.algorithm != 'nn':
                            model.pickle_model()
                        model.store()
                        model.org_files(zip_only=True)

                    # Have files output to output


                # if c == 'r':  # Runs the models/featurizations for regression
                #     with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                #         print('Model Type:', alg)
                #         print('Featurization:', method)
                #         print('Dataset:', data)
                #         print()
                #         print('Initializing model...', end=' ', flush=True)
                #         # initiate model class with algorithm, dataset and target
                #         model1 = models.MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                #                                 tune=False, cv=3, opt_iter=25)
                #         print('Done.\n')
                #
                #     with cd('dataFiles'):  # Have files output to output
                #         model1.featurize()
                #         val = 0.0
                #         if alg == 'nn':
                #             val = 0.1
                #
                #         model1.data_split(val=val)
                #         model1.reg()
                #         model1.run()
                #         model1.analyze()
                #         if model1.algorithm != 'nn':
                #             model1.pickle_model()
                #
                #         model1.store()
                #         model1.org_files(zip_only=True)

#                if c == 'c':
#                     if (data == 'sider.csv' or data == 'clintox.csv') and alg == 'svc':
#                         pass
#                     else:
#                         with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
#                             print('Model Type:', alg)
#                             print('Featurization:', method)
#                             print('Dataset:', data)
#                             print('Target(s):', target)
#                             print()
#                             print('Initializing model...', end=' ', flush=True)
#                             # initiate model class with algorithm, dataset and target
#                             opt_iter=10
#                             if data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv']:
#                                 opt_iter = 25
#                             model = models.MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
#                                                    tune=False, cv=3, opt_iter=opt_iter)
#                             print('Done.\n')
#
#                         with cd('dataFiles'):
#                             # Runs classification model
#                             model.featurize()  # Featurize molecules
#                             val = 0.0
#                             if alg == 'nn':
#                                 val = 0.1
#                             model.data_split(val=val)
#                             model.reg()
#                             model.run()  # Runs the models/featurizations for classification
#                             model.analyze()
#                             if model.algorithm != 'nn':
#                                 model.pickle_model()
#                             model.store()
#                             model.org_files(zip_only=True)
#
#                         # Have files output to output


def single_model():
    """
    This model is for debugging, similiar to the lines at the bottom of models.py. This is meant
    to show how the current workflow works, as well serves as an easy spot to de-bug issues.

    :return: None
    """

    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model1 = models.MlModel(algorithm='ada', dataset='ESOL.csv', target='water-sol', feat_meth=[0, 6],
                                tune=True, cv=2, opt_iter=2)

        print('done.')
        print('Model Type:', model1.algorithm)
        print('Featurization:', model1.feat_meth)
        print('Dataset:', model1.dataset)
        print()

    with cd('output'):  # Have files output to output
        model1.featurize()
        model1.data_split(val=0.2)
        model1.reg()
        model1.run()
        model1.analyze()
        if model1.algorithm != 'nn':  # issues pickling NN models
            model1.pickle_model()
        model1.store()
        model1.org_files(zip_only=True)
        # model1.QsarDB_export(zip_output=True)
        model1.to_neo4j()


def example_load():
    """
    Example case of loading a model from a previous pickle and running an analysis on it.
    :return:
    """
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    model2 = unpickle_model('output/RE00_20200624-091038/RE00_20200624-091038.pkl')
    model2.run()
    # Make predictions
    predictions = model2.regressor.predict(model2.test_features)

    # Dataframe for replicate_model
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = model2.test_target
    pva['predicted'] = predictions
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmme = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))


if __name__ == "__main__":
      main()
#    single_model()
    # example_load()
