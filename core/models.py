'''
This code was written by Adam Luxon and team as part of the McQuade and Ferri research groups.
'''
from timeit import default_timer

from numpy.random import randint
from rdkit import RDLogger
from py2neo import Graph
from sqlalchemy.exc import OperationalError

from core.ingest import load_smiles
from core.name import name
from core.storage import MLMySqlConn
from core.neo4j import ModelToNeo4j


rds = ['lipo_raw.csv', 'ESOL.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv', 'Lipo-short.csv',
       'lipo_subset.csv']
RDLogger.DisableLog('rdApp.*')

cds = ['BBBP.csv', 'HIV.csv', 'bace.csv']

multi_label_classification_datasets = ['sider.csv', 'clintox.csv']  # List of multi-label classification data sets


class MlModel:  # TODO update documentation here
    """
    Class to set up and run machine learning algorithm.
    """
    from core.regressors import get_regressor, hyperTune, build_cnn
    from core.grid import make_grid
    from core.train import train_reg, train_cls
    from core.analysis import impgraph, pva_graph, classification_graphs, hist, plot_learning_curves
    from core.classifiers import get_classifier
    from core.storage.util import original_param
    from core.features import featurize, data_split
    from core.storage import pickle_model, store, org_files, featurize_from_mysql, QsarDB_export

    def __init__(self, algorithm, dataset, target, feat_meth, tune=False, opt_iter=10, cv=3, random=None):
        """
        Requires: learning algorithm, dataset, target property's column name, hyperparamter tune, number of
        optimization cycles for hyper tuning, and number of Cross Validation folds for tuning.
        """

        self.algorithm = algorithm
        self.dataset = dataset
        if self.algorithm in ['svm', 'ada'] and self.dataset in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv']:
            tune = False

        # Sets self.task_type based on which dataset is being used.
        if self.dataset in cds:
            self.task_type = 'single_label_classification'
        elif self.dataset in rds:
            self.task_type = 'regression'
        elif self.dataset in multi_label_classification_datasets:
            self.task_type = 'multi_label_classification'
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
            # Makes drop = False for multi-target classification
            self.data, self.smiles_series = load_smiles(self, dataset, drop=False)
        else:
            self.data, self.smiles_series = load_smiles(self, dataset)

        # define run name used to save all outputs of model
        self.run_name = name(self)

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

        if self.task_type in ['single_label_classification', 'multi_label_classification']:
            self.get_classifier()

    def run(self):
        """ Runs machine learning model. Stores results as class attributes."""

        if self.tuned:  # Do hyperparameter tuning
            self.make_grid()
            self.hyperTune(n_jobs=8)
        else:  # Return original parameter if not tuned
            self.original_param()
        # Done tuning, time to fit and predict
        if self.task_type == 'regression':
            self.train_reg()

        if self.task_type in ['single_label_classification', 'multi_label_classification']:
            self.train_cls()

    def analyze(self):
        # Variable importance for tree based estimators
                # if self.feat_meth == [0]:
        #     self.permutation_importance()
        if self.algorithm in ['rf', 'gdb', 'ada'] and self.feat_meth == [0]:
            self.impgraph()

        # make predicted vs actual graph
        if self.task_type == 'regression':
            self.pva_graph()
            self.pva_graph(use_scaled=True)  # Plot scaled pva data
            self.plot_learning_curves()
            if self.algorithm != "cnn":  # CNN is running into OverflowError: cannot convert float infinity to integer
                self.hist()

        if self.task_type in ['single_label_classification', 'multi_label_classification']:
            self.classification_graphs()

    def to_neo4j(self, port="bolt://localhost:7687", username="neo4j", password="password"):
        # Create Neo4j graphs from pipeline
        t1 = default_timer()
        self.neo4j_params = {'port': port, 'username': username, 'password': password}  # Pass Neo4j Parameters
        Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Test connection to Neo4j
        ModelToNeo4j(model=self, port=port, molecules_per_batch=5000, username=username, password=password)
        t2 = default_timer() - t1
        print(f"Time it takes to finish graphing {self.run_name}: {t2}sec")

    def connect_mysql(self, user, password, host, database, initialize_all_data=False):
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
        if initialize_all_data:
            conn.insert_all_data_mysql()
