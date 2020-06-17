"""
This code was written by Adam Luxon and team as part of the McQuade research group.
"""
import csv
import os
import subprocess
import shutil
import pathlib

from numpy.random import randint

from core import ingest, misc, name
from core.features import featurize, data_split  # imported function becomes instance method
from core.regressors import get_regressor, hyperTune
from core.train import train_reg, train_cls
from core.analysis import impgraph, pva_graph
from core.classifiers import get_classifier
from core.storage import export_json, pickle_model, unpickle_model

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning."""

        self.predictions = None
        self.predictions_stats = None
        self.impgraph = None
        self.varimp = None
        self.pva_graph = None

        self.hyper_tuned = False

        self.algorithm = algorithm
        self.dataset = dataset

        # Sets self.task_type based on which dataset is being used.
        if self.dataset == 'sider.csv':
            self.task_type = 'classification'
        elif self.dataset == 'Lipophilicity-ID.csv' or self.dataset == 'ESOL.csv' or self.dataset == 'water-energy.csv' or self.dataset == 'logP14k.csv' or self.dataset == 'jak2_pic50.csv':
            self.task_type = 'regression'

        self.target_name = target
        self.feat_meth = feat_meth

        if random is None:
            self.random_seed = randint(low=1, high=50)
        else:
            self.random_seed = random

        self.opt_iter = opt_iter
        self.cv_folds = cv
        self.tune = tune

        # ingest data.  collect full data frame (self.data)
        # collect pandas series of the SMILES (self.smiles_col)
        self.data, self.smiles_series = ingest.load_smiles(self, dataset)

        if self.task_type == 'regression':
            self.regressor, self.task_type = get_regressor(self)

        if self.task_type == 'classification':
            self.regressor, self.task_type = get_classifier(self)

        self.run_name = name.name(self.algorithm, self.dataset,
                                  self.feat_meth, self.tune)  # Create file nameprint(dict(vars(model1)).keys())

        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None
            self.regressor = self.regressor()  # make it callable to match Tune = True case
            self.tune_time = None

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        if self.tune != self.hyper_tuned:
            raise Exception("If tune=False, cannot hypertune model")

        self.run_name = name.name(self.algorithm, self.dataset, self.feat_meth,
                                  self.tune)  # Create file nameprint(dict(vars(model1)).keys())
        # TODO naming scheme currently must be called after featurization declared--adjust for robust

        # Done tuning, time to fit and predict
        if self.task_type == 'regression':
            self.predictions, self.predictions_stats = train_reg(self)

        if self.task_type == 'classification':
            self.predictions, self.predictions_stats = train_cls(self)

    def analyze(self, output):

        with misc.cd(output):

            # # Variable importance for rf and gdb
            if self.algorithm in ['rf', 'gdb', 'rfc'] and self.feat_meth == [0]:
                self.varimp = impgraph(self)

            # make predicted vs actual graph
            # self.pva_graph = pva_graph(self)
            pva_graph(self)
            # TODO Make classification graphing function


    def store(self, output):
        """  Organize and store model inputs and outputs.  """

        # create model file name
        csvfile = ''.join("%s.csv" % self.run_name)

        with misc.cd(output):
            # create dictionary of attributes
            att = dict(vars(self))  # makes copy so does not affect original attributes
            del att['data']  # do not want DF in dict
            # del att['smiles']  # do not want series in dict
            # del att['graphM']  # do not want graph object
            # del att['stats']  # will unpack and add on
            # del att['pvaM']  # do not want DF in dict
            del att['run_name']
            try:
                del att['varimp']  # don't need variable importance in our machine learning results record
                del att['impgraph']  # Don't need a graph object in our csv
            except KeyError:
                pass
            # del att['impgraph']
            # att.update(self.stats)
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
            # self.pvaM.to_csv(''.join("%s_predictions.csv" % self.run_name))

            # save graphs
            # self.graphM.savefig(''.join("%s_PvAM" % self.run_name), transparent=True)
            # if self.algorithm in ['rf', 'gdb'] and self.feat_meth == [0]:
            #     self.impgraph.savefig(''.join("%s_impgraph" % self.run_name), transparent=True)
            #     self.impgraph.close()
            # else:
            #     pass
            # self.graphM.close()  # close to conserve memory when running many models.
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
# with misc.cd(str(pathlib.Path(__file__).parent.parent.absolute()) + '/dataFiles/'):
#     print('Now in:', os.getcwd())
#     print('Initializing model...', end=' ', flush=True)
#     # initiate model class with algorithm, dataset and target
#     model1 = MlModel(algorithm='rf', dataset='ESOL.csv', target='water-sol', feat_meth=[0],
#                      tune=False, cv=3, opt_iter=25)
#     print('done.')
#
# # # featurize data with rdkit2d
# model1 = featurize(model1)
# model1 = data_split(model1, val=0.0)
# # model1 = hyperTune(model1)
#
# run_name = model1.run_name
# pickle_model(model=model1, run_name=run_name)
# model2 = unpickle_model(run_name=run_name)
#
# model2.run()
# model2.analyze(output=str(pathlib.Path(__file__).parent.parent.absolute()) + '/output/')
# model2.store(output=str(pathlib.Path(__file__).parent.parent.absolute()) + '/output/')
#
# pickle_model(model=model2, run_name=run_name)
# model3 = unpickle_model(run_name=run_name)
#
# model3.run()
# export_json(model1)


# import pprint
# print(dict(vars(model1)).keys())
# pprint.pprint(dict(vars(model1)))
# # print(model1.feat_meth)
