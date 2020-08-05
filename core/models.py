'''
This code was written by Adam Luxon and team as part of the McQuade and Ferri research groups.
'''
from core import ingest
from core.storage.mysql_storage import MLMySqlConn
from numpy.random import randint
from core import name

from core.neo4j.nodes_to_neo4j import nodes
from core.neo4j.rel_to_neo4j import relationships
from rdkit import RDLogger
from timeit import default_timer
from py2neo import Graph
from sqlalchemy.exc import OperationalError


rds = ['Lipophilicity-ID.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv', 'Lipo-short.csv']
RDLogger.DisableLog('rdApp.*')

cds = ['BBBP.csv', 'HIV.csv', 'bace.csv']

multi_label_classification_datasets = ['sider.csv', 'clintox.csv']  # List of multi-label classification data sets


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from core.regressors import get_regressor, hyperTune
    from core.grid import make_grid
    from core.train import train_reg, train_cls
    from core.analysis import impgraph, pva_graph
    from core.classifiers import get_classifier

    from core.features import featurize, data_split
    from core.storage.storage import pickle_model, store, org_files
    from core.storage.mysql_storage import featurize_from_mysql
    from core.storage.qsardq_export import QsarDB_export

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """
        Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning.
        """

        self.algorithm = algorithm
        self.dataset = dataset

        # Sets self.task_type based on which dataset is being used.
        if self.dataset in cds or self.dataset in multi_label_classification_datasets:
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
            self.data, self.smiles_series = ingest.load_smiles(self, dataset,
                                                               drop=False)  # Makes drop = False for multi-target classification
        else:
            self.data, self.smiles_series = ingest.load_smiles(self, dataset)

        # define run name used to save all outputs of model
        self.run_name = name.name(self)

        if not tune:  # if no tuning, no optimization iterations or CV folds.
            self.opt_iter = None
            self.cv_folds = None
            self.tune_time = None

        self.mysql_params = None
        self.neo4j_params = None

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
        # if self.feat_meth == [0]:
        #     self.permutation_importance()
        if self.algorithm in ['rf', 'gdb', 'rfc', 'ada'] and self.feat_meth == [0]:
            self.impgraph()

        # make predicted vs actual graph
        self.pva_graph()
        # TODO Make classification graphing function

    def to_neo4j(self, port, username, password):
        # Create Neo4j graphs from pipeline

        time_dict = {'Base Nodes': None, 'Molecules': None, 'Fragments': None, 'Features': None,
                     'Misc relationships': None, 'Total Time': None, 'Number of Molecules': None,
                     'Number of Fragments': None}

        t1 = default_timer()
        self.neo4j_params = {'port': port, 'username': username, 'password': password}  # Pass Neo4j Parameters
        Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Test connection to Neo4j
        time_dict = nodes(self, time_dict)  # Create nodes
        time_dict = relationships(self, time_dict)  # Create relationships
        t2 = default_timer() - t1
        time_dict['Total Time'] = t2
        # print(f"Time it takes to finish graphing {self.run_name}: {t2}sec")
        return time_dict

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
            conn.insert_data_mysql()
