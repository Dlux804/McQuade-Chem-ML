'''
This code was written by Adam Luxon and team as part of the McQuade research group.
'''
from core import ingest, features, grid, regressors, analysis, name, misc
# from main import ROOT_DIR
import csv
import os
import subprocess


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from features import featurize  # imported function becomes instance method
    # from .misc import foo

    def __init__(self, algorithm, dataset, target, drop=True):
        """Requires: learning algorithm, dataset and target property's column name."""
        self.algorithm = algorithm
        self.dataset = dataset
        self.target_name = target
        # ingest data.  collect full data frame (self.data)
        # collect pandas series of the SMILES (self.smiles_col)
        self.data, self.smiles_col = ingest.load_smiles(self,dataset, drop)



    # # this is unnecessary function.  Just make features.featurize an import and have it access/store to self.
    # def featurization(self, feats=None):
    #     """ Featurize molecules in dataset and stores results as attribute in class instance.
    #         Keyword arguments:
    #         feats -- Features you want.  Default = None (requires user input)
    #     """
    #     self.data, self.feat_meth, self.feat_time = features.featurize(self.data, self.algorithm, feats)




    def run(self, tune=False):
        """ Runs machine learning model. Stores results as class attributes."""

        # store tune as attribute for cataloguing
        self.tuned = tune

        # Split data up. Set random seed here for graph comparison purposes.
        # TODO Collect and store the molecules in train, test and validation data sets
        if self.algorithm == 'nn':
            train_features, test_features, val_features, train_target, test_target, val_target,  self.feature_list = features.targets_features(self, val=0.1, random=42)

        else:
            train_features, test_features, train_target, test_target, self.feature_list = features.targets_features(
            self.data, self.target, val=0.1, random=42)

        # store the shape of the input data
        self.in_shape = train_features.shape[1]
        print("shape:", self.in_shape)
        # set the model specific regressor function from sklearn
        self.regressor = regressors.regressor(self.algorithm)

        if tune:  # Do hyperparameter tuning

            # ask for tuning variables (not recommended for high throughput)
            # folds = int(input('Please state the number of folds for hyperparameter searching: '))
            # iters = int(input('Please state the number of iterations for hyperparameter searching: '))
            # jobs = int(input('Input the number of processing cores to use. (-1) to use all.'))

            # FIXME Unfortunate hard code deep in the program.
            folds = 2
            iters = 3
            jobs = -1  # for bayes, max jobs = folds.

            # Make parameter grid
            param_grid = grid.make_grid(self.algorithm)

            # Run Hyper Tuning
            params,  self.tuneTime = regressors.hyperTune(self.regressor(), train_features,
                                                                train_target, param_grid, folds, iters, jobs=folds)

            # redefine regressor model with best parameters.
            self.regressor = self.regressor(**params)  # **dict will unpack a dictionary for use as keywrdargs

        else:  # Don't tune.
            self.regressor = self.regressor()  # make it callable to match Tune = True case
            self.tuneTime = None

        # Done tuning, time to fit and predict

        #Variable importance for rf and gdb
        if self.algorithm in ['rf', 'gdb'] and self.feat_meth == [0]:
            self.impgraph, self.varimp = analysis.impgraph(self.algorithm, self.regressor, train_features, train_target, self.feature_list)
        else:
            pass
        # multipredict
        # self.pvaM, fits_time = analysis.multipredict(self.regressor, train_features, test_features, train_target, test_target)
        self.stats, self.pvaM, fits_time = analysis.replicate_multi(self.regressor, train_features, test_features, train_target, test_target)

        self.graphM = analysis.pvaM_graphs(self.pvaM)

        # run the model 5 times and collect the metric stats as dictionary
        # self.stats = analysis.replicate_model(self, 5)

    def store(self):
        """  Organize and store model inputs and outputs.  """

        # create model file name
        run_name = name.name(self.algorithm, self.dataset, self.feat_meth, self.tuned)
        csvfile = ''.join("%s.csv" % run_name)

        # create dictionary of attributes
        att = dict(vars(self))  # makes copy so does not affect original attributes
        del att['data']  # do not want DF in dict
        del att['smiles']  # do not want series in dict
        del att['graphM']  # do not want graph object
        del att['stats']  # will unpack and add on
        # del att['impgraph']
        att.update(self.stats)
        att.update(self.varimp)
        # Write contents of attributes dictionary to a CSV
        with open(csvfile, 'w') as f:  # Just use 'w' mode in Python 3.x
            w = csv.DictWriter(f, att.keys())
            w.writeheader()
            w.writerow(att)
            f.close()

        # save data frames
        self.data.to_csv(''.join("%s_data.csv" % run_name))
        self.pvaM.to_csv(''.join("%s_predictions.csv" % run_name))

        # save graphs
        self.graphM.savefig("%s_PvAM" % run_name)
        if self.algorithm in ['rf', 'gdb'] and self.feat_meth == [0]:
            self.impgraph.savefig("%s_impgraph" % run_name)
            self.impgraph.close()
        else:
            pass
        self.graphM.close()  # close to conserve memory when running many models.
        # self.graph.savefig(name+'PvA')

        # make folders for each run
        try:
            os.mkdir(run_name)
        except OSError as e:
            pass

        # put output files into new folder
        filesp = 'mv ./' + run_name + '* ' + run_name +'/'
        subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

        # Move folder to output/
        # when testing using code below, need ../output/ because it will run from core.
        # when running from main.py at root, no ../ needed.
        movesp = 'mv ./' + run_name + '/ output/'

        subprocess.Popen(movesp, shell=True, stdout=subprocess.PIPE)  # run bash command



# This section is for troubleshooting and should be commented out when finished testing

# change active directory
with misc.cd('../dataFiles/'):
    print('Now in:', os.getcwd())
    print('Initializing model...', end=' ', flush=True)
    # initiate model class with algorithm, dataset and target
    model1 = MlModel('nn', 'ESOL.csv', 'water-sol')
    print('done.')

# # featurize data with rdkit2d
model1.featurize([0])
# # print(model1.feat_meth)
#
#
# #
# # # Run the model with hyperparameter optimization
model1.run(tune=False)
# print("Input shape: ", model1.in_shape)
#
# print('Tune Time:', model1.tuneTime)
#



# Save results
# model1.store()
#
#
# # Must show() graph AFTER it has been saved.
# # if show() is called before save, the save will be blank
# # display PvA graph
# model1.graphM.show()

