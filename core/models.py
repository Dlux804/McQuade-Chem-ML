'''
This code was written by Adam Luxon and team as part of the McQuade and Ferri research groups.
'''
from core import ingest
from core.storage.mysql_storage import MLMySqlConn
from numpy.random import randint
from core import name
from core.neo4j import nodes, relationships
from rdkit import RDLogger
from py2neo import Graph
import time

from core import featurize, featurize_from_mysql, data_split, pickle_model, store, org_files, QsarDB_export
from core.regressors import get_regressor, hyperTune
from core.classifiers import get_classifier
from core.grid import make_grid
from core.train import train_reg, train_cls
from core.analysis import impgraph, pva_graph

from sqlalchemy.exc import OperationalError

RDLogger.DisableLog('rdApp.*')

# g = Graph("bolt://localhost:7687", user="neo4j", password="1234")

rds = ['Lipophilicity-ID.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv', '18k-logP.csv']
cds = ['sider.csv', 'clintox.csv', 'BBBP.csv', 'HIV.csv', 'bace.csv']


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """
        Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning.
        """

        self.algorithm = algorithm
        self.dataset = dataset
        self.target_name = target
        self.feat_meth = feat_meth
        self.tuned = tune
        self.opt_iter = opt_iter
        self.cv_folds = cv

        self.multi_label_classification_datasets = ['sider.csv', 'clintox.csv']  # multi-label classification datasets

        # Sets self.task_type based on which dataset is being used.
        if self.dataset in cds:
            self.task_type = 'classification'
            # Tuning not working for classification
            if self.tuned:
                print(f"Not tuning running tuning on classification model, currently broken")
                self.tuned = False
        elif self.dataset in rds:
            self.task_type = 'regression'
        else:
            raise Exception(f'{self.dataset} is an unknown dataset, cannot choose classification or regression')

        # Define random seed
        if random is None:
            self.random_seed = randint(low=1, high=50)
        else:
            self.random_seed = random

        # ingest data, collect full data frame (self.data)
        self.data = ingest.load_smiles(self, dataset)

        # define run name used to save all outputs of model
        self.run_name = name.name(self)

        # if no tuning, no optimization iterations or CV folds.
        if not self.tuned:
            self.opt_iter = None
            self.cv_folds = None
            self.tune_time = None

        # Define loose variables
        self.mysql_params = None
        self.neo4j_params = None

    def connect_mysql(self, user, password, host, database):
        # Gather MySql Parameters
        self.mysql_params = {'user': user, 'password': password, 'host': host, 'database': database}

        # Test connection
        try:
            conn = MLMySqlConn(user=self.mysql_params['user'], password=self.mysql_params['password'],
                               host=self.mysql_params['host'], database=self.mysql_params['database'])
        except OperationalError:
            return Exception("Bad parameters passed to connect to MySql database or MySql server"
                             "not properly configured")
        # Test if dataset feat_meth combo exist, if not create it
        conn.init_table(self.dataset, self.feat_meth)

    def featurize(self, retrieve_from_mysql=False):
        if retrieve_from_mysql:
            self.data, self.feat_time, self.feat_method_name = featurize_from_mysql(self)
        else:
            self.data, self.feat_time, self.feat_method_name = featurize(self)

    def data_split(self, test=0.2, val=0):
        (self.data, self.scaler, self.in_shape, self.n_tot,
         self.n_train, self.n_test, self.n_val, self.target_name, self.target_array,
         self.feature_array, self.feature_list, self.feature_length,
         self.train_features, self.test_features, self.val_features,
         self.train_percent, self.test_percent, self.val_percent,
         self.train_target, self.test_target, self.val_target,
         self.train_molecules, self.test_molecules, self.val_molecules) = data_split(self, test, val)

    def reg(self):  # broke this out because input shape is needed for NN regressor to be defined.
        """
        Function to fetch regressor.  Should be called after featurization has occured and input shape defined.
        :return:
        """

        if self.task_type == 'regression':
            self.regressor, self.fit_params = get_regressor(self, call=False)  # returns instantiated model estimator

        if self.task_type == 'classification':
            self.regressor = get_classifier(self)

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        if self.tuned:  # Do hyperparameter tuning
            self.param_grid = make_grid(self)
            (self.tune_algorithm_name, self.cp_delta,
             self.cp_n_best, self.params,
             self.regressor, self.fit_params, self.tune_time) = hyperTune(self, n_jobs=8)

        # Done tuning, time to fit and predict
        if self.task_type == 'regression':
            self.predictions, self.predictions_stats = train_reg(self)

        if self.task_type == 'classification':
            self.predictions, self.predictions_stats = train_cls(self)

    def analyze(self):
        # Variable importance for tree based estimators
        if self.algorithm in ['rf', 'gdb', 'rfc'] and self.feat_meth == [0]:
            self.varimp = impgraph(self)

        # make predicted vs actual graph
        if self.dataset not in self.multi_label_classification_datasets:
            pva_graph(self)
        # TODO Make classification graphing function

    def pickle_model(self):
        pickle_model(self)

    def store(self):
        store(self)

    def org_files(self, zip_only=False):
        org_files(self, zip_only)

    def QsarDB_export(self, zip_output=False):
        QsarDB_export(self, zip_output)

    def to_neo4j(self, port, username, password):
        # Create Neo4j graphs from pipeline
        t1 = time.perf_counter()
        self.neo4j_params = {'port': port, 'username': username, 'password': password}  # Pass Neo4j Parameters
        Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Test connection to Neo4j
        nodes(self)  # Create nodes
        relationships(self)  # Create relationships
        t2 = time.perf_counter()
        print(f"Time it takes to finish graphing {self.run_name}: {t2 - t1}sec")
