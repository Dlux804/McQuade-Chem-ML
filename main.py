# TODO: Make main function that asks user what models they would like to initiate

from core import models
from core.misc import cd
from core.storage import pickle_model, unpickle_model
import os
import pathlib
from core.features import featurize

# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # Asking the user to decide between running classification models or regression models
    c = str(input("Enter c for classification, r for regression: "))

    # Sets up learner, featurizations, and datasets for classification.
    if c == 'c':
        # list of available classification learning algorithms
        learner = ['svc', 'knc', 'rfc']

        # list of available featurization methods

        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4],
                 [5]]  # Change this to change which featurizations are being tested (for classification)

        # classification data sets in dict. Key: Filename.csv , Value: Target column header
        sets = {
            'sider.csv': 'Injury, poisoning and procedural complications'
        }

    # Sets up learner, featurizations, and data sets for regression
    if c == 'r':
        # list of available regression learning algorithms
        learner = ['ada', 'rf', 'svr', 'gdb', 'mlp', 'knn']

        # list of available featurization methods
        feats = [[0], [0, 2], [0, 3], [0, 4], [0, 5], [2], [3], [4],
                 [5]]  # Change this to change which featurizations are being tested (for regression)

        # regression data sets in dict. Key: Filename.csv , Value: Target column header
        sets = {
            'Lipophilicity-ID.csv': 'exp',
            'ESOL.csv': 'water-sol',
            'water-energy.csv': 'expt',
            'logP14k.csv': 'Kow',
            'jak2_pic50.csv': 'pIC50'
        }

    for alg in learner:  # loop over all learning algorithms

        for method in feats:  # loop over the featurization methods

            for data, target in sets.items():  # loop over dataset dictionary
                if c == 'r':  # Runs the models/featurizations for regression
                    with cd('dataFiles'):
                        print('Now in:', os.getcwd())
                        print('Initializing model...', end=' ', flush=True)

                        # initiate model class with algorithm, dataset and target
                        model = models.MlModel(alg, data, target, method)
                        print('done.')

                    print('Model Type:', alg)
                    print('Featurization:', method)
                    print('Dataset:', data)
                    print()

                    # run model
                    model.featurize()  # Featurize molecules
                    model.data_split(val=0.0)  # Split molecules
                    model.run()  # Bayes Opt
                    pickle_model(model, file_location=f'{model.run_name}.pkl')
                    model.analyze()  # Runs analysis on model
                    # save results of model
                    model.store()
                    # save results in json format
                    model.export_json()

                if c == 'c':  # Runs the models/featurizations for classification
                    # change active directory
                    with cd('dataFiles'):
                        print('Now in:', os.getcwd())
                        print('Initializing model...', end=' ', flush=True)

                        # initiate model class with algorithm, dataset and target
                        model = models.MlModel(alg, data, target, method)
                        print('done.')

                    print('Model Type:', alg)
                    print('Featurization:', method)
                    print('Dataset:', data)
                    print()
                    # Runs classification model
                    model.featurize()  # Featurize molecules
                    model.run()


def example_model():
    """
    This model is for debugging, similiar to the lines at the bottom of models.py. This is meant
    to show how the current workflow works, as well serves as an easy spot to de-bug issues.

    :return: None
    """

    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model = models.MlModel(algorithm='rf', dataset='water-energy.csv', target='expt', feat_meth=[0],
                                tune=False, cv=3, opt_iter=5)
        print('done.')

    model.featurize()
    model.data_split(val=0.0)
    model.run()
    pickle_model(model, file_location=f'{model.run_name}.pkl')
    model.store()
    model.export_json()


if __name__ == "__main__":
    # main()
    example_model()
