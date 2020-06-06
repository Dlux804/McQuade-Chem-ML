'''
This code was written by Adam Luxon and team as part of the McQuade research group.
'''
from core import ingest, features, grid, regressors, analysis, name, misc, classifiers
import csv
import os
import subprocess
import shutil
from numpy.random import randint
from core import name


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from features import featurize, data_split  # imported function becomes instance method
    from regressors import get_regressor, hyperTune
    from grid import make_grid
    from train import train_reg, train_cls
    from analysis import impgraph, pva_graph
    from classifiers import get_classifier
    from storage import export_json
    # from .misc import foo

    def __init__(self, algorithm, dataset,  target, task, tune=False, opt_iter=10, cv=3):
        """Requires: learning algorithm, dataset and target property's column name."""
        self.algorithm = algorithm
        self.dataset = dataset
        self.target_name = target
        self.task_type = task  # regression or classification
        # TODO Make task identificaiton automatic based on dataset.
        # ingest data.  collect full data frame (self.data)
        # collect pandas series of the SMILES (self.smiles_col)
        self.data, self.smiles_series = ingest.load_smiles(self, dataset)
        self.random_seed = randint(low=1, high=50)
        self.opt_iter = opt_iter
        self.cv_folds = cv
        self.tuned = tune
        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None


    # def featurization(self, feats=None):
    #     """ Featurize molecules in dataset and stores results as attribute in class instance.
    #         Keyword arguments:
    #         feats -- Features you want.  Default = None (requires user input)
    #     """
    #     self.data, self.feat_meth, self.feat_time = features.featurize(self.data, self.algorithm, feats)

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        self.run_name = name.name(self.algorithm, self.dataset, self.feat_meth, self.tuned)  # Create file nameprint(dict(vars(model1)).keys())
        if self.task_type == 'regression':
            self.get_regressor()

        if self.task_type == 'classification':
            self.get_classifier()

        if self.tuned:  # Do hyperparameter tuning
            self.make_grid()
            self.hyperTune(n_jobs =6)

        else:  # Don't tune.
            self.regressor = self.regressor()  # make it callable to match Tune = True case
            self.tune_time = None


        # Done tuning, time to fit and predict
        if self.task_type == 'regression':
            self.train_reg()

        if self.task_type == 'classification':
            self.train_cls()

    def analyze(self):
        # # Variable importance for rf and gdb
        if self.algorithm in ['rf', 'gdb', 'rfc'] and self.feat_meth == [0]:
            self.impgraph()

        # make predicted vs actual graph
        self.pva_graph()

        # else:
        #     pass
        # # multipredict
        # # self.pvaM, fits_time = analysis.multipredict(self.regressor, train_features, test_features, train_target, test_target)
        # self.stats, self.pvaM, fits_time = analysis.replicate_multi(self.regressor, train_features, test_features, train_target, test_target)
        # self.graphM = analysis.pvaM_graphs(self.pvaM)

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
        # filesp = ''.join(['move ./', self.run_name, '* ', self.run_name, '/'])  # move for Windows system
        filesp = ''.join(['mv ./', self.run_name, '* ', self.run_name, '/'])  # mv for Linux system
        subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

        # movepkl = ''.join(['move ./', '.pkl', '* ', self.run_name, '/'])  # move for Windows system
        movepkl = ''.join(['mv ./', '.pkl', '* ', self.run_name, '/']) # mv for Linux system
        subprocess.Popen(movepkl, shell=True, stdout=subprocess.PIPE)  # run bash command

        # Move folder to output/
        # when testing using code below, need ../output/ because it will run from core.
        # when running from main.py at root, no ../ needed.
        # movesp = 'move ./' + run_name + ' output/'
        #
        # subprocess.Popen(movesp, shell=True, stdout=subprocess.PIPE)  # run bash command


    def classification_run(self):
        # set the model specific classifier function from sklearn
        self.classifier = classifiers.classifier(self.algorithm)

        x_train, x_test, y_train, y_test, feature_list = features.targets_features(self.data,self.target) # splits the data into a training set and a test set
        random_model = self.classifier()
        random_model.fit(x_train, y_train)

        random_prediction = random_model.predict(x_test)

        accuracy, confusion, classification, roc_auc_score = analysis.classification_metrics(random_prediction, y_test) # Evaluates model's performance on the test data
        print()     # Formatting for terminal
        print('Accuracy_score: ')
        print(accuracy)
        print('Confusion_matrix: ')
        print(confusion)
        print('Classification_report: ')
        print(classification)
        print('roc_auc_score: ')
        print(roc_auc_score)



# This section is for troubleshooting and should be commented out when finished testing

# change active directory
with misc.cd('../dataFiles/'):
    print('Now in:', os.getcwd())
    print('Initializing model...', end=' ', flush=True)
    # initiate model class with algorithm, dataset and target
    model1 = MlModel('rf', 'ESOL.csv', 'water-sol', 'regression', tune=True, cv=3, opt_iter=25)
    print('done.')

# # featurize data with rdkit2d
model1.featurize([0])
model1.data_split(val=0.0)
model1.run()
model1.analyze()
model1.export_json()
import pprint


# print(dict(vars(model1)).keys())
# pprint.pprint(dict(vars(model1)))
# # print(model1.feat_meth)
#
#
# #
# # # Run the model with hyperparameter optimization
# model1.run(tune=False)
# print("Input shape: ", model1.in_shape)
#
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

