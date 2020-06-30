'''
This code was written by Adam Luxon and team as part of the McQuade and Ferri research groups.
'''
from core import ingest, features, grid, regressors, analysis, name, misc, classifiers
import csv
import os
import subprocess
import shutil
from numpy.random import randint
from core import name

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

rds = ['Lipophilicity-ID.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv']
cds = ['sider.csv']


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from core.features import featurize, data_split  # imported function becomes instance method
    from core.regressors import get_regressor, hyperTune
    from core.grid import make_grid
    from core.train import train_reg, train_cls
    from core.analysis import impgraph, pva_graph
    from core.classifiers import get_classifier
    from core.storage import export_json, org_files, pickle_model, unpickle_model

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning."""

        self.algorithm = algorithm
        self.dataset = dataset

        # Sets self.task_type based on which dataset is being used.
        if self.dataset in cds:
            self.task_type = 'classification'
        elif self.dataset in rds:
            self.task_type = 'regression'
        else:
            raise Exception('{} is an unknown dataset! Cannot choose classification or regression.'.format(self.dataset))

        self.target_name = target
        self.feat_meth = feat_meth

        if random is None:
            self.random_seed = randint(low=1, high=50)
        else:
            self.random_seed = random

        self.opt_iter = opt_iter
        self.cv_folds = cv
        self.tuned = tune

        # ingest data.  collect full data frame (self.data)
        # collect pandas series of the SMILES (self.smiles_col)
        self.data, self.smiles_series = ingest.load_smiles(self, dataset)

        # get the specified learning algorithm


        self.run_name = name.name(self.algorithm, self.dataset, self.feat_meth,
                                  self.tuned)  # Create file nameprint(dict(vars(model1)).keys())

        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None
            # self.regressor = self.regressor()  # make it callable to match Tune = True case
            self.tune_time = None

    def reg(self):  # broke this out because input shape is needed for NN regressor to be defined.
        """
        Function to fetch regressor.  Should be called after featurization has occured and input shape defined.
        :return:
        """
        if self.task_type == 'regression':
            self.get_regressor(call=False)

        if self.task_type == 'classification':
            self.get_classifier()

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        # TODO naming scheme currently must be called after featurization declared--adjust for robust

        if self.tuned:  # Do hyperparameter tuning
            self.make_grid()
            self.hyperTune(n_jobs=8)

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
        # TODO Make classification graphing function
