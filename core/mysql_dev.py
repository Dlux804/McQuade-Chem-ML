from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
import pymysql  # Hidden import, please leave this
import pandas as pd
import os
from tqdm import tqdm

from core.storage import compress_fingerprint

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

conn = create_engine('mysql+pymysql://user:Lookout@10@localhost/test', pool_recycle=3600)

feat_sets = {0: 'rdkit2d', 1: 'rdkit2dnormalized', 2: 'rdkitfpbits', 3: 'morgan3counts',
             4: 'morganfeature3counts', 5: 'morganchiral3counts', 6: 'atompaircounts'}


def retrieve_data(dataset, feat_meth=None, feat_name=None):
    if feat_meth is None and feat_name is None:
        raise Exception("Must specify a feature method (as an number) or feature name")
    elif feat_meth is not None and feat_name is not None:
        raise Exception("Can only specify a feature method (as a number) or a feature name")
    elif feat_meth is not None:
        sql_data_table = f'{dataset}_{feat_sets[feat_meth]}'
    elif feat_name is not None:
        sql_data_table = f'{dataset}_{feat_name}'

    try:
        return pd.read_sql(f"select * from test.`{sql_data_table}`;", conn)
    except ProgrammingError:
        return None


def insert_data_mysql():

    bad_datasets = ['cmc.csv']

    datasets_dir = '/home/user/PycharmProjects/McQuade-Chem-ML/dataFiles'
    datasets = os.listdir(datasets_dir)
    for dataset in datasets:
        for feat_id, feat_name in feat_sets.items():

            print(feat_name, dataset)
            if dataset not in bad_datasets and retrieve_data(dataset=dataset, feat_name=feat_name) is None:

                data = pd.read_csv(datasets_dir + '/' + dataset)
                generator = MakeGenerator([feat_name])

                columns = []
                for name, numpy_type in generator.GetColumns():
                    columns.append(name)

                smi = tqdm(data['smiles'], desc="Featurization")
                features = map(generator.process, smi)
                features = pd.DataFrame(features, columns=columns)
                data = pd.concat([data, features], axis=1)
                data = data.drop(list(data.filter(regex='_calculated')), axis=1)
                data = data.drop(list(data.filter(regex='[lL]og[pP]')), axis=1)
                data = data.dropna(axis=1)

                data = compress_fingerprint(data)

                data.to_sql(f'{dataset}_{feat_name}', conn, if_exists='fail')
                print(f'Created {dataset}_{feat_name}')


# retrieve_data('', '')
insert_data_mysql()
