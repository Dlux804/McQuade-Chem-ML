'''
This code was written by Adam Luxon and team as part of the McQuade and Ferri research groups.
'''
import time

from rdkit import RDLogger
from py2neo import Graph
from numpy.random import randint
from sqlalchemy.exc import OperationalError

from core import load_smiles, get_run_name, featurize, data_split
from core.storage import MLMySqlConn, pickle_model, store, QsarDB_export, org_files, featurize_from_mysql
from core.neo4j import nodes, relationships


RDLogger.DisableLog('rdApp.*')

# g = Graph("bolt://localhost:7687", user="neo4j", password="1234")

rds = ['Lipophilicity-ID.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv', '18k-logP.csv']
cds = ['sider.csv', 'clintox.csv', 'BBBP.csv', 'HIV.csv', 'bace.csv']


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from core.regressors import get_regressor, hyperTune
    from core.grid import make_grid
    from core.train import train_reg, train_cls
    from core.analysis import impgraph, pva_graph
    from core.classifiers import get_classifier
    from core.storage import initialize_tables

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """
        Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning.
        """

        self.algorithm = algorithm
        self.dataset = dataset
        multi_label_classification_datasets = ['sider.csv',
                                               'clintox.csv']  # List of multi-label classification data sets
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

        if self.dataset in multi_label_classification_datasets:
            self.data, self.smiles_series = load_smiles(self, dataset, drop=False)
        else:
            self.data, self.smiles_series = load_smiles(self, dataset)

        # define run name used to save all outputs of model
        self.run_name = get_run_name(self)

        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None
            self.tune_time = None

        self.mysql_params = None
        self.neo4j_params = None

    def connect_mysql(self, user, password, host, database, initialize_data=False):
        # Gather MySql Parameters
        self.mysql_params = {'user': user, 'password': password, 'host': host, 'database': database}

        # Test connection
        try:
            conn = MLMySqlConn(user=self.mysql_params['user'], password=self.mysql_params['password'],
                               host=self.mysql_params['host'], database=self.mysql_params['database'])
        except OperationalError:
            return Exception("Bad parameters passed to connect to MySql database or MySql server"
                             "not properly configured")

        # Insert featurized data into MySql. This will only run once per dataset/feat combo,
        # even if initialize_data=True
        if initialize_data:
            self.initialize_tables()

    def featurize(self, retrieve_from_mysql=False):
        if retrieve_from_mysql:
            featurize_from_mysql(self)
        else:
            featurize(self, retrieve_from_mysql=retrieve_from_mysql)

    def data_split(self, test=0.2, val=0):
        data_split(self, test, val)

    def reg(self):  # broke this out because input shape is needed for NN regressor to be defined.
        """
        Function to fetch regressor.  Should be called after featurization has occured and input shape defined.
        :return:
        """
        if self.task_type == 'regression':
            self.get_regressor(call=False)  # returns instantiated model estimator

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
        self.pva_graph()
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
