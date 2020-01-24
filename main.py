
# TODO: Make main function that asks user what models they would like to initiate

from core import models
from core.misc import cd
import os

# Creating a global variable to be imported from all other models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    print('ROOT Working Directory:', ROOT_DIR)

    # list of available learning algorithms
    # learner = ['ada', 'rf', 'svr', 'gdb', 'mlp', 'knn']
    # learner = ['gdb', 'rf', 'ada', 'knn']
    learner = ['gdb']

    # list of available featurization methods
    # featurize = [[0], [0, 2], [0, 3], [0, 4], [0,5], [0, 6], [2], [3], [4], [5], [6]]
    # featurize = [[0], [2], [3], [4], [0, 2], [0, 3], [0, 4]]
    featurize = [[0]]
    # features for models that require normalized data (nn, svm)
    # norm_featurize = [[1], [1,2], [1,3], [1,4], [1,5], [1,6], [2], [3], [4], [5], [6]]
    norm_featurize = [[1], [2], [3], [4], [1, 2], [1, 3], [1, 4]]

    # data sets in dict. Key: Filename.csv , Value: Target column header
    # sets = {
    #     'Lipophilicity-ID.csv': 'exp',
    #     'ESOL.csv': 'water-sol',
    #     'water-energy.csv': 'expt',
    #     'logP14k.csv': 'Kow',
    #     'jak2_pic50.csv': 'pIC50'
    # }
    sets = {
        'water-energy.csv': 'expt'
    }
    for alg in learner:  # loop over all learning algorithms

        if alg == 'mlp' or alg == 'svr': # if the algorithm needs normalized data
            feats = norm_featurize
        else:
            feats = featurize

        for method in feats:  # loop over the featurization methods

            for data, target in sets.items():  # loop over dataset dictionary

                # change active directory
                with cd('dataFiles'):

                    print('Now in:', os.getcwd())
                    print('Initializing model...', end=' ', flush=True)

                    # initiate model class with algorithm, dataset and target
                    model = models.MlModel(alg, data, target)
                    print('done.')

                print('Model Type:', alg)
                print('Featurization:', method)
                print('Dataset:', data)
                print()
                # featurize molecules
                model.featurization(method)

                # run model
                model.run(tune=True) # Bayes Opt

                # save results of model
                model.store()




    # # Initiate Model
    # model1 = MlModel('ada', 'ESOL.csv', 'water-sol')
    #
    # # featurize data with rdkit2d
    # model1.featurization([0])
    # print(model1.feat_meth)
    #
    # # Run the model with hyperparameter optimization
    # model1.run(tune=True)
    # # print('Tune Time:', model1.tuneTime)
    #
    # # Save results
    # model1.store()

if __name__ == "__main__":
    main()

