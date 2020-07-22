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
from core.nodes_to_neo4j import nodes
from core.rel_to_neo4j import relationships
from core.fragments import fragments_to_neo
from rdkit import RDLogger
from py2neo import Graph
import time

RDLogger.DisableLog('rdApp.*')

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


rds = ['Lipophilicity-ID.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv']
cds = ['sider.csv', 'clintox.csv', 'BBBP.csv', 'HIV.csv', 'bace.csv']


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
    from core.storage import store, org_files, pickle_model, unpickle_model
    from core.qsardq_export import QsarDB_export

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """
        Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning.
        """

        self.algorithm = algorithm
        self.dataset = dataset
        self.multi_label_classification_datasets = ['sider.csv', 'clintox.csv'] # List of multi-label classification data sets
        # Sets self.task_type based on which dataset is being used.
        if self.dataset in cds:
            self.task_type = 'classification'
        elif self.dataset in rds:
            self.task_type = 'regression'
        else:
            raise Exception(
                '{} is an unknown dataset! Cannot choose classification or regression.'.format(self.dataset))

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

        if self.dataset in self.multi_label_classification_datasets:
            self.data, self.smiles_series = ingest.load_smiles(self, dataset, drop = False) # Makes drop = False for multi-target classification
        else:
            self.data, self.smiles_series = ingest.load_smiles(self, dataset)

        # define run name used to save all outputs of model
        self.run_name = name.name(self)

        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None
            self.tune_time = None

    def reg(self):  # broke this out because input shape is needed for NN regressor to be defined.
        """
        Function to fetch regressor.  Should be called after featurization has occured and input shape defined.
        :return:
        """
        if self.task_type == 'regression':
            self.get_regressor(call=False) # returns instantiated model estimator

        if self.task_type == 'classification':
            self.get_classifier()

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        if self.tuned:  # Do hyperparameter tuning
            self.make_grid()
            self.hyperTune(n_jobs=8)

        # Done tuning, time to fit and predict
        if self.task_type == 'regression':
            self.train_reg()

        if self.task_type == 'classification':
            self.train_cls()

    def analyze(self):
        # Variable importance for tree based estimators
        if self.algorithm in ['rf', 'gdb', 'rfc'] and self.feat_meth == [0]:
            self.impgraph()

        # make predicted vs actual graph
        if self.dataset in rds:
            self.pva_graph()
        # TODO Make classification graphing function

    def to_neo4j(self):
        # Create Neo4j graphs from pipeline
        t1 = time.perf_counter()
        nodes(self)  # Create nodes
        relationships(self)  # Create relationships
        t2 = time.perf_counter()
        print(f"Time it takes to finish graphing {self.run_name}: {t2-t1}sec")