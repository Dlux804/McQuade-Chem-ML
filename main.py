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

    #### list of all learning algorithms
    learner = ['svm','rf', 'gdb', 'nn']

    #### All tune option
    tune_option = [False, True]

    #### Features
    feats = [[0], [1], [2], [3], [4], [5], [0,1]]  # Use this line to select specific featurizations

    #### classification data sets for reference/testing
    # sets = {
    #     'BBBP.csv': targets,
    # 'sider.csv': targets,
    # 'clintox.csv': targets,
    # 'bace.csv': targets,
    #    }

    #### regression data sets for reference/testing
    # sets = {
    #     'ESOL.csv': 'water-sol',
    #     'lipo_raw.csv': 'exp',
    #     'water-energy.csv': 'expt',
    #     'logP14k.csv': 'Kow',
    #     'jak2_pic50.csv': 'pIC50',
    #     'Lipophilicity-ID.csv': 'exp'
    # }
    sets = {'Lipophilicity-ID.csv': 'exp'}

    #### Split percent
    test_percents = [0.1, 0.2, 0.3, 0.4]

    #### Data Splitting methods
    splitters = ['random', 'index', 'scaffold']

    #### Data scaling methods
    scalers = ['standard', 'minmax', None]

    #### Tuning methods
    tuners = ["bayes", 'random', 'grid']


    for alg in learner:  # loop over all learning algorithms

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
                    for test_percent in test_percents:
                        for splitter in splitters:
                            for scale in scalers:
                                for tune in tuners:
                                    with cd(str(
                                            pathlib.Path(
                                                __file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                                        print('Model Type:', alg)
                                        print('Featurization:', method)
                                        print('Dataset:', data)
                                        print('Target(s):', target)
                                        print('Task type:', task_type)
                                        print()
                                        print('Initializing model...', end=' ', flush=True)
                                        # initiate model class with algorithm, dataset and target

                                        model = MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                                                        tune=tune, cv=2, opt_iter=2)
                                        print('Done.\n')
                                        model.featurize()
                                        if model.algorithm not in ["nn", "cnn"]:
                                            model.data_split(split=splitter, test=test_percent, scaler=scale)
                                        else:
                                            model.data_split(split="scaffold", test=test_percent, val=0.1, scaler="standard")
                                    with cd('output'):
                                        model.reg()
                                        model.run(tuner=tune)  # Runs the models/featurizations for classification
                                        # model.analyze()
                                        # if model.algorithm not in ['nn', 'cnn']:
                                        #     model.pickle_model()
                                        model.store()
                                        model.org_files(zip_only=True)
                                        model.to_neo4j(port="bolt://localhost:7687", username="neo4j",
                                                       password="password")
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
        # model3 = MlModel(algorithm='gdb', dataset='water-energy.csv', target='expt', feat_meth=[0],
        #                  tune=False, cv=2, opt_iter=2)
        model3 = MlModel(algorithm='gdb', dataset='logP14k.csv', target='Kow', feat_meth=[0],
                         tune=False, cv=2, opt_iter=2)
        print('done.')
        print('Model Type:', model3.algorithm)
        print('Featurization:', model3.feat_meth)
        print('Dataset:', model3.dataset)
        print()
        model3.featurize()
        if model3.algorithm not in ["nn", "cnn"]:
            model3.data_split(split="index", test=0.1, scaler="standard")
        else:
            model3.data_split(split="scaffold", test=0.1, val=0.1, scaler="standard")
    with cd('output'):  # Have files output to output
        model3.reg()
        model3.run(tuner="bayes")
        # model3.analyze()
        # if model3.algorithm != 'nn':  # issues pickling NN models
        #     model3.pickle_model()
        model3.store()
        model3.org_files(zip_only=True)
        # model1.QsarDB_export(zip_output=True)
        model3.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")


def example_run_with_mysql_and_neo4j(dataset='logP14k.csv', target='Kow'):
    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model3 = MlModel(algorithm='cnn', dataset=dataset, target=target, feat_meth=[0, 2, 3],
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


# def split_test():
#     with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
#         print('Now in:', os.getcwd())
#         print('Initializing model...', end=' ', flush=True)
#         # initiate model class with algorithm, dataset and target
#         model3 = MlModel(algorithm='rf', dataset='water-energy.csv', target='expt', feat_meth=[0],
#                          tune=False, cv=2, opt_iter=2)
#         print('done.')
#         print('Model Type:', model3.algorithm)
#         print('Featurization:', model3.feat_meth)
#         print('Dataset:', model3.dataset)
#         print()
#         model3.featurize()
#         if model3.algorithm not in ["nn", "cnn"]:
#             model3.data_split(split="scaffold", test=0.1, scaler="standard",
#                               add_molecule_to_testset=["CN(C)C(=O)c1ccc(cc1)OC", "CS(=O)(=O)Cl"])
#         else:
#             model3.data_split(split="scaffold", test=0.1, val=0.1, scaler="standard")
#     with cd('output'):  # Have files output to output
#         model3.reg()
#         model3.run(tuner="random")
#         model3.store()
#         model3.org_files(zip_only=True)
#         model3.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")


if __name__ == "__main__":
    main()
    # single_model()
    # split_test()
    # deepchem_split()
    # example_load()
    # example_run_with_mysql_and_neo4j()
    # Qsar_import_examples()
    # output_dir_to_neo4j()
    # QsarToNeo4j('2012ECM185.zip')
