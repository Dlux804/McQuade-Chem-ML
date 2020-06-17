
# TODO: Make main function that asks user what models they would like to initiate

import pathlib
import os

from core import models
from core.misc import cd
from core.features import featurize, data_split  # imported function becomes instance method
from core.storage import export_json, pickle_model, unpickle_model
from core.regressors import hyperTune

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

        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4], [5]] # Change this to change which featurizations are being tested (for classification)

        # classification data sets in dict. Key: Filename.csv , Value: Target column header
        sets = {
            'sider.csv': 'Injury, poisoning and procedural complications'
        }

    # Sets up learner, featurizations, and data sets for regression
    else:  # c == 'r'
        # list of available regression learning algorithms
        learner = ['ada', 'rf', 'svr', 'gdb', 'mlp', 'knn']

        # list of available featurization methods
        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4], [5]] # Change this to change which featurizations are being tested (for regression)

        # regression data sets in dict. Key: Filename.csv , Value: Target column header
        sets = {
            'Lipophilicity-ID.csv': 'exp',
            'ESOL.csv': 'water-sol',
            'water-energy.csv': 'expt',
            'logP14k.csv': 'Kow',
            'jak2_pic50.csv': 'pIC50'
        }

    for alg in learner: # loop over all learning algorithms

        for method in feats:  # loop over the featurization methods

            for data, target in sets.items():  # loop over dataset dictionary
                if c == 'r':     # Runs the models/featurizations for regression
                    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):

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
                    model = featurize(model) # Featurize molecules
                    model.run()  # Bayes Opt
                    model.analyze() # Runs analysis on model
                    # save results of model
                    model.store()

                if c == 'c':     # Runs the models/featurizations for classification
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
                    model = featurize(model)  # Featurize molecules
                    model.run()


def example_model():
    """
    This model is for debugging, similiar to the lines at the bottom of models.py. This is meant
    to show how the current workflow works, as well serves as an easy spot to de-bug issues.

    :return: None
    """
    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Create model object
        print('Now in:', os.getcwd())
        print('Initializing model...', end=' ', flush=True)
        # initiate model class with algorithm, dataset and target
        model1 = models.MlModel(algorithm='rf', dataset='ESOL.csv', target='water-sol', feat_meth=[0],
                         tune=True, cv=3, opt_iter=25)
        print('done.')

    # # featurize data with rdkit2d
    model1 = featurize(model1)
    model1 = data_split(model1, val=0.0)
    model1 = hyperTune(model1)  # hyperTune is pulled out of models.py

    run_name = model1.run_name  # Demo pickle before running model
    pickle_model(model=model1, run_name='output/' + run_name)
    model2 = unpickle_model(run_name='output/' + run_name)

    model2.run()
    model2.analyze(output=str(pathlib.Path(__file__).parent.absolute()) + '/output')
    model2.store(output=str(pathlib.Path(__file__).parent.absolute()) + '/output')
    # export_json(model2)  # Currently broken

    pickle_model(model=model2, run_name='output/' + run_name)  # Demo pickle after running and analyzing data
    model3 = unpickle_model(run_name='output/' + run_name)
    model3.run()


if __name__ == "__main__":
    # main()
    example_model()
