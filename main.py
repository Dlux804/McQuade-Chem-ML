# TODO: Make main function that asks user what models they would like to initiate

from core import models, Get_Classification
from core.storage.misc import cd
from core.storage.storage import pickle_model, unpickle_model
import os
import pathlib
from time import sleep
from core.features import featurize
from core.neo4j.output_to_neo4j import output_to_neo4j

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
        learner = ['svc', 'knc', 'rf']
        #learner = [] # Use this line to test specific models instead of iterating


        targets = None
        sets = {
            'BBBP.csv': targets,
            'sider.csv': targets,
            'clintox.csv': targets,
            'bace.csv': targets,
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


                if c == 'r':  # Runs the models/featurizations for regression
                    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                        print('Model Type:', alg)
                        print('Featurization:', method)
                        print('Dataset:', data)
                        print()
                        print('Initializing model...', end=' ', flush=True)
                        # initiate model class with algorithm, dataset and target
                        model1 = models.MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                                                tune=False, cv=3, opt_iter=25)
                        print('Done.\n')

                    with cd('dataFiles'):  # Have files output to output
                        model1.featurize()
                        val = 0.0
                        if alg == 'nn':
                            val = 0.1

                        model1.data_split(val=val)
                        model1.reg()
                        model1.run()
                        model1.analyze()
                        if model1.algorithm != 'nn':
                            model1.pickle_model()

                        model1.store()
                        model1.org_files(zip_only=True)

                if c == 'c':
                    targets = Get_Classification.get_classification_targets(data)  # Gets targets for classification based on the data set being used

                    if (data == 'sider.csv' or data == 'clintox.csv') and alg == 'svc':
                        pass
                    else:
                        # change active directory
                        with cd('dataFiles'):
                            print('Now in:', os.getcwd())
                            print('Initializing model...', end=' ', flush=True)

                            # initiate model class with algorithm, dataset and target
                            model = models.MlModel(alg, data, targets, method)
                            print('done.')

                        print('Model Type:', alg)
                        print('Featurization:', method)
                        print('Dataset:', data)
                        print('Target(s):', targets)
                        print()
                        # Runs classification model
                        model.featurize()  # Featurize molecules
                        model.data_split()
                        model.reg()
                        model.run()  # Runs the models/featurizations for classification


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
        model1.QsarDB_export(zip_output=True)


def example_run_with_mysql_and_neo4j():
    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model3 = models.MlModel(algorithm='gdb', dataset='water-energy.csv', target='expt', feat_meth=[0, 4],
                                tune=True, cv=2, opt_iter=2)
        print('done.')
        print('Model Type:', model3.algorithm)
        print('Featurization:', model3.feat_meth)
        print('Dataset:', model3.dataset)
        print()

    with cd('output'):  # Have files output to output
        model3.connect_mysql(user='user', password='Lookout@10', host='localhost', database='featurized_databases',
                             initialize_data=True)
        model3.featurize(retrieve_from_mysql=True)
        model3.data_split(val=0.1)
        model3.reg()
        model3.run()
        model3.analyze()
        if model3.algorithm != 'nn':  # issues pickling NN models
            model3.pickle_model()

        model3.store()
        model3.org_files(zip_only=True)
        # model1.QsarDB_export(zip_output=True)
        model3.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")


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
    # main()
    single_model()
    # example_load()
    # example_run_with_mysql_and_neo4j()
    # output_to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")

    from core import MLMySqlConn
    conn = MLMySqlConn(user='neo4j', password='password', host='localhost', database='featurized_databases')
    """
    This serves as an example for how to import different classes into current file (Notice how mysql_storag.py
    was never imported directly). 
    """
