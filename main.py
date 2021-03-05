# TODO: Make main function that asks user what models they would like to initiate
import os
import pathlib
from core import MlModel, get_classification_targets, Get_Task_Type_1
from core.storage import cd


# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    #### list of all learning algorithms
    learner = ['rf','nn', 'svm',  'gdb', 'nn']

    #### All tune option
    tune_option = [False, True]

    #### Features
    feats = [[0], [1], [2], [3], [4], [5], [0,1], [0,2], [0,3], [0,4], [0,5]]  # Use this line to select specific featurizations

    sets = {
        'flashpoint.csv': 'flashpoint',
        'logP14k.csv': 'Kow',
        'jak2_pic50.csv': 'pIC50',
        'Lipophilicity-ID.csv': 'exp',
        'ESOL.csv': 'water-sol',
        'water-energy.csv': 'expt'
    }

    #### Split percent
    test_percents = [0.2, 0.3]

    #### Data Splitting methods
    splitters = ['random', 'index', 'scaffold']

    #### Data scaling methods
    scalers = ['standard', 'minmax', None]

    #### Tuning methods
    tuners = ['bayes',  'random']  # "grid",

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
                    for isTune in tune_option:
                        for test_percent in test_percents:
                            for splitter in splitters:
                                for scale in scalers:
                                    for tuner in tuners:
                                        with cd(str(pathlib.Path(
                                                __file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                                            print('Model Type:', alg)
                                            print('Featurization:', method)
                                            print('Dataset:', data)
                                            print('Target(s):', target)
                                            print('Task type:', task_type)
                                            print('Tuning:', isTune)
                                            print('Tuner:', tuner)
                                            print()
                                            print('Initializing model...', end=' ', flush=True)
                                            # initiate model class with algorithm, dataset and target

                                            model = MlModel(algorithm=alg, dataset=data, target=target,
                                                            feat_meth=method, tune=isTune, cv=5, opt_iter=25)
                                            print('Done.\n')
                                            model.connect_mysql(user='user', password='dolphin', host='localhost',
                                                                 database='featurized_datasets',
                                                                 initialize_all_data=False)
                                            model.featurize(retrieve_from_mysql=True)
                                            if model.algorithm not in ["nn", "cnn"]:
                                                model.data_split(split=splitter, test=test_percent, scaler=scale)
                                            else:
                                                model.data_split(split=splitter, test=test_percent, val=0.1,
                                                                 scaler=scale)
                                        with cd('output'):
                                            model.reg()
                                            model.run(tuner=tuner)  # Runs the models/featurizations for classification
                                            model.analyze()
                                            if model.algorithm not in ['nn', 'cnn']:
                                                model.pickle_model()
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
        model3 = MlModel(algorithm='gdb', dataset='Lipophilicity-ID.csv', target='exp', feat_meth=[0],
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

        model3.store()
        model3.org_files(zip_only=True)
        model3.to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")







if __name__ == "__main__":
    main()

