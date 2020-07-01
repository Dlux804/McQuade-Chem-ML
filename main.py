# TODO: Make main function that asks user what models they would like to initiate

from core import models
from core.misc import cd
from core.storage import pickle_model, unpickle_model
import os
import pathlib
from time import sleep
from core.features import featurize

# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # Asking the user to decide between running classification models or regression models
    # c = str(input("Enter c for classification, r for regression: "))
    c = 'r'

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
        learner = ['ada', 'rf', 'svr', 'gdb', 'nn', 'knn']
        learner = ['rf', 'gdb', 'nn']


        # list of available featurization methods
        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4],
                 [5]]
        feats = [[0], [0, 2]]  # Change this to change which featurizations are being tested (for regression)

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


                    with cd(str(pathlib.Path(__file__).parent.absolute()) + '/dataFiles/'):  # Initialize model
                        print('Model Type:', alg)
                        print('Featurization:', method)
                        print('Dataset:', data)
                        print()
                        print('Initializing model...', end=' ', flush=True)
                        # initiate model class with algorithm, dataset and target
                        model1 = models.MlModel(algorithm=alg, dataset=data, target=target, feat_meth=method,
                                                tune=False, cv=3, opt_iter=5)
                        print('Done.\n')

                    with cd('output'):  # Have files output to output
                        model1.featurize()
                        if alg == 'nn':
                            val = 0.1
                        else:
                            val = 0.0
                        model1.data_split(val=val)
                        model1.reg()
                        model1.run()
                        model1.analyze()
                        if model1.algorithm != 'nn':
                            model1.pickle_model()

                        model1.export_json()
                        model1.org_files(zip_only=True)

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
        model1 = models.MlModel(algorithm='nn', dataset='logP14k.csv', target='Kow', feat_meth=[0, 2],
                                tune=True, cv=10, opt_iter=50)
        print('done.')

    with cd('output'):  # Have files output to output
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        model1.analyze()
        if model1.algorithm != 'nn':
            model1.pickle_model()

        # pickle_model(model1, file_location='dev.pkl')  # Create pickled model for faster testing

        # model1 = unpickle_model(file_location='dev.pkl
        model1.export_json()
        model1.org_files(zip_only=True)
        # sleep(5)  # give org_files() time to move things, otherwise zip fails.
        # model1.zip_files()

        # model1.store()

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
    predictions = model2.regressor.predict(self.test_features)

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