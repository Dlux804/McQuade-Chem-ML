
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

    # Sets up learner and datasets for classification.
    if c == 'c':
        # list of available classification learning algorithms
        learner = ['svc', 'knc', 'rf']
        #learner = ['rf'] # Use this line to test specific models instead of iterating

        datasets = ['sider.csv', 'clintox.csv', 'BBBP.csv', 'HIV.csv', 'bace.csv']
        #datasets = ['sider.csv'] # Use this line to test specific data sets instead of having to iterate

    # Sets up learner, featurizations, and data sets for regression
    if c == 'r':
        # list of available regression learning algorithms
        learner = ['ada', 'rf', 'svr', 'gdb', 'mlp', 'knn']

        # list of available featurization methods
        feats = [[0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1], [2], [3], [4],
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
        # The following if statements set featurization options based on if the
        # model needs normalized data (currently only set up for the classification models)
        if alg == 'svc':   # Normalized
            feats = [[1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1], [2], [3], [4],
                 [5], [6]]
        if alg == 'knc':   # Normalized
            feats = [[1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1], [2], [3], [4],
                     [5], [6]]
        if alg == 'rf':    # Not Normalized
            feats = [[0], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0], [2], [3], [4],
                 [5], [6]]
        for method in feats:  # loop over the featurization methods

                if c == 'r':  # Runs the models/featurizations for regression
                    for data, target in sets.items():
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
                        model.data_split()
                        model.run()  # Bayes Opt

                        model.analyze()  # Runs analysis on model
                        # save results of model
                        model.store()
                        # loop over dataset dictionary


                if c == 'c':
                    for data in datasets:

                        # The following if statements allow for multi-label classification by iterating through each target column depending on the data set. Please provide feedback for ways to make this more intuitive and less cluttered.
                        if data == 'sider.csv':
                            targets = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Gastrointestinal disorders',
                                       'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                                       'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                                       'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                                       'Psychiatric disorders', 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders',
                                       'Nervous system disorders', 'Injury, poisoning and procedural complications']
                        if data == 'clintox.csv':
                            targets = ['FDA_APPROVED', 'CT_TOX']
                        if data == 'BBBP.csv':
                            targets = ['p_np']
                        if data == 'HIV.csv':
                            targets = ['HIV_active']
                        if data == 'bace.csv':
                            targets = ['Class']

                        for target in targets:   # Loops through different targets for each data set (multi target classification)
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
                            print('Target:', target)
                            print()
                            # Runs classification model
                            model.featurize()  # Featurize molecules
                            model.data_split()
                            model.run()  # Runs the models/featurizations for classification


                        # loop over dataset dictionary



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
        model1 = models.MlModel(algorithm='rf', dataset='ESOL.csv', target='water-sol', feat_meth=[0],
                         tune=False, cv=3, opt_iter=25)
        print('done.')

    with cd('output'):  # Have files output to output
        model1.featurize()
        model1.data_split(val=0.0)
        model1.run()
        model1.analyze()
        pickle_model(model1, file_location='dev.pkl')  # Create pickled model for faster testing

        model1 = unpickle_model(file_location='dev.pkl')
        model1.export_json()
        model1.store()


if __name__ == "__main__":
    main()
    # example_model()
