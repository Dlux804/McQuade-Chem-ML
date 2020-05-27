'''
This code was written by Adam Luxon and team as part of the McQuade research group.
'''
from core import ingest, features, grid, regressors, analysis
# from main import ROOT_DIR
import csv
import os
import subprocess
import shutil


class MlModel:
    """
    Class to set up and run machine learning algorithm.
    """
    def __init__(self, algorithm, dataset, target, drop=True):
        """Requires: learning algorithm, dataset and target property's column name."""
        self.algorithm = algorithm
        self.dataset = dataset
        self.target = target
        self.data, self.smiles = ingest.load_smiles(self,dataset, drop)

    def featurization(self, feats=None):
        """ Featurize molecules in dataset and stores results as attribute in class instance.
            Keyword arguments:
            feats -- Features you want.  Default = None (requires user input)
        """
        self.data, self.feat_meth, self.feat_time = features.featurize(self.data, self.algorithm, feats)


    def run(self, tune=False):
        """ Runs machine learning model. Stores results as class attributes."""

        # store tune as attribute for cataloguing
        self.tuned = tune

        # Split data up. Set random seed here for graph comparison purposes.
        train_features, test_features, train_target, test_target, self.feature_list = features.targets_features(self.data, self.target, random=42)

        # set the model specific regressor function from sklearn
        self.regressor = regressors.regressor(self.algorithm)

        if tune:  # Do hyperparameter tuning

            # ask for tuning variables (not recommended for high throughput)
            # folds = int(input('Please state the number of folds for hyperparameter searching: '))
            # iters = int(input('Please state the number of iterations for hyperparameter searching: '))
            # jobs = int(input('Input the number of processing cores to use. (-1) to use all.'))

            # FIXME Unfortunate hard code deep in the program.
            folds = 2
            iters = 50
            jobs = -1  # for bayes, max jobs = folds.

            # Make parameter grid
            param_grid = grid.make_grid(self.algorithm)

            # Run Hyper Tuning
            self.params,  self.tuneTime, self.run_name = regressors.hyperTune(self.regressor(), self.algorithm, self.dataset,
                                                                         self.feat_meth, self.tuned, train_features,
                                                                         train_target, param_grid, folds, iters,
                                                                         jobs=folds)

            # redefine regressor model with best parameters.
            self.regressor = self.regressor(**self.params)  # **dict will unpack a dictionary for use as keywrdargs

        else:  # Don't tune.
            self.regressor = self.regressor()  # make it callable to match Tune = True case
            self.tuneTime = None

        # Done tuning, time to fit and predict

        # Variable importance for rf and gdb
        if self.algorithm in ['rf', 'gdb'] and self.feat_meth == [0]:
            self.impgraph, self.varimp = analysis.impgraph(self.algorithm, self.regressor, train_features, train_target, self.feature_list)
        else:
            pass
        # multipredict
        # self.pvaM, fits_time = analysis.multipredict(self.regressor, train_features, test_features, train_target, test_target)
        self.stats, self.pvaM, fits_time = analysis.replicate_multi(self.regressor, train_features, test_features, train_target, test_target)
        self.graphM = analysis.pvaM_graphs(self.pvaM)

        self.regressor = self.params

    def store(self):
        """  Organize and store model inputs and outputs.  """

        # create model file name

        csvfile = ''.join("%s.csv" % self.run_name)

        try:
            os.mkdir(self.run_name)
        except OSError as e:
            pass

        # create dictionary of attributes
        att = dict(vars(self))  # makes copy so does not affect original attributes
        del att['data']  # do not want DF in dict
        del att['smiles']  # do not want series in dict
        del att['graphM']  # do not want graph object
        del att['stats']  # will unpack and add on
        del att['pvaM']  # do not want DF in dict
        del att['run_name']
        del att['params']
        try:
            del att['varimp']  # don't need variable importance in our machine learning results record
            del att['impgraph']  # Don't need a graph object in our csv
        except KeyError:
            pass
        # del att['impgraph']
        att.update(self.stats)
        # att.update(self.varimp)
        # Write contents of attributes dictionary to a CSV
        with open(csvfile, 'w') as f:  # Just use 'w' mode in Python 3.x
            w = csv.DictWriter(f, att.keys())
            w.writeheader()
            w.writerow(att)
            f.close()

        # Path to output directory
        output_directory = ''.join(['%s\output' % os.getcwd()])
        # Copy csv file to ouput directory
        shutil.copy(csvfile, output_directory)

        # save data frames
        self.data.to_csv(''.join("%s_data.csv" % self.run_name))
        self.pvaM.to_csv(''.join("%s_predictions.csv" % self.run_name))

        # save graphs
        self.graphM.savefig(''.join("%s_PvAM" % self.run_name), transparent=True)
        if self.algorithm in ['rf', 'gdb'] and self.feat_meth == [0]:
            self.impgraph.savefig(''.join("%s_impgraph" % self.run_name), transparent=True)
            self.impgraph.close()
        else:
            pass
        self.graphM.close()  # close to conserve memory when running many models.
        # self.graph.savefig(name+'PvA')

        # make folders for each run
        # put output files into new folder
        filesp = ''.join(['move ./', self.run_name, '* ', self.run_name, '/'])  # move for Windows system
        # filesp = ''.join(['mv ./', self.run_name, '* ', self.run_name, '/'])  # mv for Linux system
        subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

        movepkl = ''.join(['move ./', '.pkl', '* ', self.run_name, '/'])  # move for Windows system
        # movepkl = ''.join(['mv ./', '.pkl', '* ', self.run_name, '/']) # mv for Linux system
        subprocess.Popen(movepkl, shell=True, stdout=subprocess.PIPE)  # run bash command

        # Move folder to output/
        # when testing using code below, need ../output/ because it will run from core.
        # when running from main.py at root, no ../ needed.
        # movesp = 'move ./' + run_name + ' output/'
        #
        # subprocess.Popen(movesp, shell=True, stdout=subprocess.PIPE)  # run bash command



# This section is for troubleshooting and should be commented out when finished testing

# change active directory
# with misc.cd('../dataFiles/'):
#     print('Now in:', os.getcwd())
#     print('Initializing model...', end=' ', flush=True)
#     # initiate model class with algorithm, dataset and target
#     model1 = MlModel('rf', 'ESOL.csv', 'water-sol')
#     print('done.')

# # featurize data with rdkit2d
# model1.featurization([0])
# print(model1.feat_meth)
#
#
# # Run the model with hyperparameter optimization
# model1.run(tune=True)

# print('Tune Time:', model1.tuneTime)

#
#
#
# # Save results
# model1.store()
#
#
# # Must show() graph AFTER it has been saved.
# # if show() is called before save, the save will be blank
# # display PvA graph
# model1.graphM.show()

