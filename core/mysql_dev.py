from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
import pymysql  # Hidden import, please leave this
import pandas as pd
import os

from core.storage import compress_fingerprint

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MLMySqlConn:
    from core.features import featurize

    def __init__(self, user, password, host, database, pool_recycle=None):
        if pool_recycle is None:
            self.conn = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        else:
            self.conn = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}',
                                      pool_recycle={pool_recycle})
        self.feat_sets = {0: 'rdkit2d', 1: 'rdkit2dnormalized', 2: 'rdkitfpbits', 3: 'morgan3counts',
                          4: 'morganfeature3counts', 5: 'morganchiral3counts', 6: 'atompaircounts'}
        self.data = None
        self.feat_meth = None
        self.database = database

    def retrieve_data(self, dataset, feat_meth=None, feat_name=None):
        # Get table name from dataset and feature method passed
        if feat_meth is None and feat_name is None:
            raise Exception("Must specify a feature method (as an number) or feature name")
        elif feat_meth is not None and feat_name is not None:
            raise Exception("Can only specify a feature method (as a number) or a feature name")
        elif feat_meth is not None:
            sql_data_table = f'{dataset}_{self.feat_sets[feat_meth]}'
        else:  # feat_name is not None
            sql_data_table = f'{dataset}_{feat_name}'

        try:
            fetched_df = pd.read_sql(f"select * from `{self.database}`.`{sql_data_table}`;", self.conn)

            return fetched_df
        except ProgrammingError:
            return None

    def insert_data_mysql(self):
        """
        The point of this is to featurize and insert all the featurized data into MySql from datafiles.
        :return:
        """

        bad_datasets = ['cmc.csv']  # This dataset seems to be giving me a hard time

        datasets_dir = '/home/user/PycharmProjects/McQuade-Chem-ML/dataFiles'
        datasets = os.listdir(datasets_dir)
        for dataset in datasets:
            for feat_id, feat_name in self.feat_sets.items():
                print(feat_name, dataset)
                if dataset not in bad_datasets and self.retrieve_data(dataset=dataset, feat_name=feat_name) is None:
                    self.data = pd.read_csv(datasets_dir + '/' + dataset)
                    self.feat_meth = [feat_id]
                    self.featurize(not_silent=False)
                    self.data = compress_fingerprint(self.data)
                    self.data.to_sql(f'{dataset}_{feat_name}', self.conn, if_exists='fail')
                    print(f'Created {dataset}_{feat_name}')


dev = MLMySqlConn(user='user', password='Lookout@10', host='localhost', database='featurized_databases')
dev.insert_data_mysql()
# df = dev.retrieve_data('ESOL.csv', feat_meth=2)
