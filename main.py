# TODO: Make main function that asks user what models they would like to initiate
import os
import pathlib

import pandas as pd
from timeit import default_timer

from core import MlModel, get_classification_targets, Get_Task_Type_1
from core.storage import cd, pickle_model, unpickle_model, QsarToNeo4j

from core.neo4j import ModelToNeo4j

# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # list of all learning algorithms
    learner = ['knn', 'rf', 'ada', 'gdb', 'nn']
    # learner = ['knn']

    # list of available classification learning algorithms for reference/testing
    # learner = ['svm', 'knn', 'rf']

    # list of available regression learning algorithms for reference/testing
    # learner = ['ada', 'rf', 'svm', 'gdb', 'nn', 'knn']

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
    # 'sider.csv': targets,
    # 'clintox.csv': targets,
    # 'bace.csv': targets,
    #    }

    # regression data sets for reference/testing
    # sets = {
    #     'ESOL.csv': 'water-sol',
    #     'Lipophilicity-ID.csv': 'exp',
    #     'water-energy.csv': 'expt',
    #     'logP14k.csv': 'Kow',
    #     'jak2_pic50.csv': 'pIC50'
    # }

    # sets = {'water-energy.csv': 'expt'}

    for alg in learner:  # loop over all learning algorithms
        # feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3],
        #          [4], [5], [0, 1, 2]]  # Use this line to select specific featurizations
        feats = [[0]]
        for method in feats:  # loop over the featurization methods
            for data, target in sets.items():  # loop over dataset dictionary

                # This gets the target columns for classification data sets (Using target lists in the dictionary causes errors later in the workflow)
                if data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv']:
                    target = get_classification_targets(data)

                # This checker allows for main.py to skip over algorithm/data set combinations that are not compatible.
                checker, task_type = Get_Task_Type_1(data, alg)
                if checker == 0:
                    pass
                else:
                    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                        print('Model Type:', alg)
                        print('Featurization:', method)
                        print('Dataset:', data)
                        print('Target(s):', target)
                        print('Task type:', task_type)
                        print()
                        print('Initializing model...', end=' ', flush=True)
                        # initiate model class with algorithm, dataset and target

                        model = MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                                        tune=False, cv=2, opt_iter=2)
                        print('Done.\n')

                    with cd('output'):
                        model.featurize()  # Featurize molecules
                        val = 0.0
                        if alg == 'nn':
                            val = 0.1
                        model.data_split(val=val)
                        model.reg()
                        model.run()  # Runs the models/featurizations for classification
                        # model.analyze()
                        if model.algorithm != 'nn':
                            model.pickle_model()
                        model.store()
                        model.org_files(zip_only=True)
                        # model.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")
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
        # model1 = MlModel(algorithm='rf', dataset='BBBP.csv', target=get_classification_targets(data='BBBP.csv'), feat_meth=[0],
        #                  tune=False, cv=2, opt_iter=5, random=10)
        model1 = MlModel(algorithm='cnn', dataset='water-energy.csv', target='expt', feat_meth=[0],
                         tune=True, cv=2, opt_iter=50)
        # model1 = MlModel(algorithm='rf', dataset='clintox.csv', target=['FDA_APPROVED', 'CT_TOX'], feat_meth=[0],
        #                  tune=False, cv=2, opt_iter=2)

        print('done.')
        print('Model Type:', model1.algorithm)
        print('Featurization:', model1.feat_meth)
        print('Dataset:', model1.dataset)
        print()
    with cd('output'):  # Have files output to output
        model1.featurize()
        if model1.algorithm in ['cnn', 'nn']:
            model1.data_split(val=0.1)
        else:
            model1.data_split()
        model1.reg()
        model1.run()
        model1.analyze()
        if model1.algorithm not in ['cnn', 'nn']:  # issues pickling NN models
            model1.pickle_model()
        model1.store()
        model1.org_files(zip_only=True)
        # model1.QsarDB_export(zip_output=True)
        # model1.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")


def example_run_with_mysql_and_neo4j(dataset='logP14k.csv', target='Kow'):
    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model3 = MlModel(algorithm='rf', dataset=dataset, target=target, feat_meth=[0, 2, 3],
                         tune=False, cv=2, opt_iter=2)
        print('done.')
        print('Model Type:', model3.algorithm)
        print('Featurization:', model3.feat_meth)
        print('Dataset:', model3.dataset)
        print()

    with cd('output'):  # Have files output to output
        model3.connect_mysql(user='user', password='Lookout@10', host='localhost', database='featurized_datasets',
                             initialize_all_data=False)
        model3.featurize(retrieve_from_mysql=True)
        model3.data_split(val=0.1)
        model3.reg()
        model3.run()
        # model3.analyze()
        # if model3.algorithm != 'nn':  # issues pickling NN models
        #     model3.pickle_model()

        # model3.store()
        # model3.org_files(zip_only=True)
        # model1.QsarDB_export(zip_output=True)
        start_timer = default_timer()
        # model3.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")
        return default_timer() - start_timer


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


def Qsar_import_examples():
    for directory in os.listdir('Qsar_examples'):
        directory = 'Qsar_examples/' + directory
        print(directory)
        QsarToNeo4j(directory, zipped=True)
        print(f'Passed!! {directory}')


def output_dir_to_neo4j():
    head_dir = 'output'
    for directory in os.listdir(head_dir):
        directory = head_dir + '/' + directory
        print(directory)
        ModelToNeo4j(zipped_out_dir=directory, molecules_per_batch=1000, port="bolt://localhost:7687",
                     username="neo4j", password="password")


if __name__ == "__main__":
    # main()
    # single_model()
    # example_load()
    # example_run_with_mysql_and_neo4j()
    # Qsar_import_examples()
    output_dir_to_neo4j()
    # QsarToNeo4j('2012ECM185.zip')
