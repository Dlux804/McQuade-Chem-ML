from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
import pymysql  # Hidden import, please leave this
import pandas as pd
import os
from pathlib import Path
from time import time

from core import ingest
from core.misc import cd
from core.storage import compress_fingerprint, decompress_fingerprint

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MLMySqlConn:
    from core.features import featurize

    def __init__(self, user, password, host, database, pool_recycle=None):
        if pool_recycle is None:
            self.conn = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}', pool_pre_ping=True)
        else:
            self.conn = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}',
                                      pool_recycle={pool_recycle}, pool_pre_ping=True)
        self.feat_sets = {0: 'rdkit2d', 1: 'rdkit2dnormalized', 2: 'rdkitfpbits', 3: 'morgan3counts',
                          4: 'morganfeature3counts', 5: 'morganchiral3counts', 6: 'atompaircounts',
                          -1: None}
        # Define regression datasets with their keys
        self.rds = {'Lipophilicity-ID.csv': 'exp', 'ESOL.csv': 'water-sol', 'water-energy.csv': 'expt',
                    'logP14k.csv': 'Kow', 'jak2_pic50.csv': 'pIC50', '18k-logP.csv': 'exp'}
        # Define classification datasets
        self.cds = ['sider.csv', 'clintox.csv', 'BBBP.csv', 'HIV.csv', 'bace.csv', 'cmc_noadd.csv']
        self.data = None
        self.feat_meth = None
        self.smiles_series = None
        self.target_name = None
        self.database = database

    def table_exist(self, dataset, feat_name=None):
        if feat_name is None:
            query = f"select * from `{self.database}`.`{dataset}` LIMIT 1;"
        else:
            query = f"select * from `{self.database}`.`{dataset}_{feat_name}` LIMIT 1;"
        try:
            pd.read_sql(query, self.conn, index_col='index')
            return True
        except ProgrammingError:
            return False

    def retrieve_data(self, dataset, feat_meth=None, feat_name=None):

        # Will actually retrive the table from MySql
        def __fetch_table__(database, table):
            try:
                return decompress_fingerprint(pd.read_sql(f"select * from `{database}`.`{table}`;", self.conn,
                                                          index_col='index'))
            except ProgrammingError:
                raise Exception(f"{table} does not exist in MySql database")

        # Return raw database if both feat_meth and feat_name is None
        if feat_meth is None and feat_name is None:
            sql_data_table = f'{dataset}'
            return __fetch_table__(self.database, sql_data_table)
        # Make sure that both feat_meth and feat_name are not None
        elif feat_meth is not None and feat_name is not None:
            raise Exception("Can only specify a feature method (as a number) or a feature name")

        # Check that feat_name/feat_meth is a valid variable (int, list, or None)
        if not isinstance(feat_name, (int, list, type(None))) and not isinstance(feat_meth, (int, list, type(None))):
            raise Exception("Invalid feature name(s) or feature method(s)")

        # If feat_name or feat_meth is a list, check each item in list is an int
        if isinstance(feat_name, list) and not all(isinstance(item, str) for item in feat_name):
            raise Exception("One or more invalid feature names (try using feat_meth?)")
        if isinstance(feat_meth, list) and not all(isinstance(item, int) for item in feat_meth):
            raise Exception("One or more invalid feature methods (try using feat_name?)")

        if feat_meth is not None:
            # If single feat_meth, return featurized df
            if not isinstance(feat_meth, list):
                sql_data_table = f'{dataset}_{self.feat_sets[feat_meth]}'
                return __fetch_table__(self.database, sql_data_table)
            # Convert feat_meths into feat_names
            feat_name = []
            for feat_id in feat_meth:
                feat_name.append(self.feat_sets[feat_id])

        # If single feat_name, return featurized df
        if not isinstance(feat_name, list):
            sql_data_table = f'{dataset}_{feat_meth}'
            return __fetch_table__(self.database, sql_data_table)

        # If single item in feat_name, return featurized df
        if len(feat_name) == 1:
            sql_data_table = f'{dataset}_{feat_name[0]}'
            return __fetch_table__(self.database, sql_data_table)

        # Merge multiple dataframes together
        dfs = []
        for feat in feat_name:
            sql_data_table = f'{dataset}_{feat}'
            dfs.append(__fetch_table__(self.database, sql_data_table))

        # Get the column names that are the same in the different featurized tables
        df_1_columns = list(dfs[0].columns)
        df_2_columns = list(dfs[1].columns)
        same_columns = list(set(df_1_columns) & set(df_2_columns))

        # Combine the multiple dataframes together
        data = dfs.pop(0)
        for df in dfs:
            data = data.merge(df, on=same_columns)
        return data

    def insert_data_mysql(self):
        """
        The is to featurize and insert all the featurized data into MySql from datafiles.
        :return:
        """

        bad_datasets = ['cmc.csv']  # This dataset seems to be giving me a hard time

        # Pull datasets
        datasets_dir = str(Path(__file__).parent.parent.absolute()) + '/dataFiles/'
        datasets = os.listdir(datasets_dir)
        for dataset in datasets:
            for feat_id, feat_name in self.feat_sets.items():
                # Make sure dataset and feat_meth combo is not already in MySql
                if dataset not in bad_datasets and not self.table_exist(dataset=dataset, feat_name=feat_name):
                    print(feat_name, dataset)

                    # Digest data and smiles_series
                    with cd(str(Path(__file__).parent.parent.absolute()) + '/dataFiles/'):
                        if dataset in list(self.rds.keys()):
                            self.target_name = self.rds[dataset]
                            self.data, smiles_series = ingest.load_smiles(self, dataset)
                        elif dataset in self.cds:
                            self.data, smiles_series = ingest.load_smiles(self, dataset, drop=False)
                        else:
                            raise Exception(f"Dataset {dataset} not found in rds or cds. Please list in baddies or add")

                    # Insert just the raw dataset that can be featurized (drop smiles that return None Mol objects)
                    if feat_name is None:

                        # Drop misbehaving rows
                        issue_row_list = []
                        issue_row = 0
                        for smiles in self.data['smiles']:
                            if MolFromSmiles(smiles) is None:
                                issue_row_list.append(issue_row)
                            issue_row = issue_row + 1
                        self.data.drop(self.data.index[[issue_row_list]], inplace=True)

                        # Send data to MySql
                        self.data.to_sql(f'{dataset}', self.conn, if_exists='fail')
                        print(f'Created {dataset}')

                    # Otherwise featurize data like normal
                    else:
                        self.feat_meth = [feat_id]
                        self.featurize(not_silent=False)
                        self.data = compress_fingerprint(self.data)
                        self.data.to_sql(f'{dataset}_{feat_name}', self.conn, if_exists='fail')
                        print(f'Created {dataset}_{feat_name}')


def featurize_from_mysql(self):
    # Pull data from MySql server
    if self.mysql_params is None:
        raise Exception("No connection to MySql made. Please run [model].connect_mysql()")
    start_feat = time()
    mysql_conn = MLMySqlConn(user=self.mysql_params['user'], password=self.mysql_params['password'],
                             host=self.mysql_params['host'], database=self.mysql_params['database'])
    self.data = mysql_conn.retrieve_data(dataset=self.dataset, feat_meth=self.feat_meth)
    self.feat_time = time() - start_feat
