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
    saving_i = 1
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # list of all learning algorithms
    learner = ['svc', 'knc', 'rfc', 'ada', 'rf', 'svr', 'gdb', 'nn', 'knn']
#    learner = ['rfc']

    # list of available classification learning algorithms for reference/testing
    #learner = ['svc', 'knc', 'rfc']

    # list of available regression learning algorithms for reference/testing
#    learner = ['ada', 'rf', 'svr', 'gdb', 'nn', 'knn']

    # All data sets in dict
    targets = None
    sets = {
        'BBBP.csv': targets,
        'sider.csv': targets,
        'clintox.csv': targets,
        'bace.csv': targets,
        'ESOL.csv': 'water-sol',
        'Lipophilicity-ID.csv': 'exp',
        'water-energy.csv': 'expt',
        'logP14k.csv': 'Kow',
        'jak2_pic50.csv': 'pIC50'
    }

    # classification data sets for reference/testing
    # sets = {
    #     'BBBP.csv': targets,
    #     'sider.csv': targets,
    #     'clintox.csv': targets,
    #     'bace.csv': targets,
    # }

    # regression data sets for reference/testing
    # sets = {
    #     'ESOL.csv': 'water-sol',
    #     'Lipophilicity-ID.csv': 'exp',
    #     'water-energy.csv': 'expt',
    #     'logP14k.csv': 'Kow',
    #     'jak2_pic50.csv': 'pIC50'
    # }

    for alg in learner: # loop over all learning algorithms
        feats = Get_Classification.get_classification_feats(alg) # Selects featurizations for classification based on the model being ran, if regression model; uses default featurizations
#        feats=[] # Use this line to select specific featurizations
        for method in feats:  # loop over the featurization methods
            for data, target in sets.items(): # loop over dataset dictionary
                if data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv']:
                    target = Get_Classification.get_classification_targets(data)

                if data in ['sider.csv', 'clintox.csv'] and alg == 'svc':
                    pass
                elif data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv'] and alg in ['ada', 'rf', 'svr', 'gdb', 'nn', 'knn']:
                    pass
                elif data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv'] and alg in ['svc', 'knc', 'rfc']:
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
