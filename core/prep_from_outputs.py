"""
Objective: This script stores all data from output files. The data will then be used to create nodes and
relationships based on our ontology to Neo4j
"""
from core import fragments, nodes_to_neo4j, rel_to_neo4j
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import re

# TODO ADD DOCSTRING EXPLAINING ALL VARIABLES


def split_molecules(df_from_data, split):
    """"""
    df_molecules = df_from_data.loc[df_from_data['in_set'] == split]['smiles'].tolist()
    return df_molecules


class Prep:
    """
    Objective: Class containing all the data needed for creating nodes and relationships from output files
    Intent: I wanted to find a way to contain all the necessary data in a more tidy manner and this is the only thing
            that came to my mind. I didn't put comments explaining each of them since I think they are pretty
            self-explanatory.
    Note: I have to put "num_feature_list" (number of features) in a list because that's the only way
            I can obtain the number of features in a clean manner. If I use "len" on the Series object, it will only
            return 1 instead the correct number of features. Turning the Series object into numpy array also return
            1 for "len". Seems like "len" is counting the number of element in a Pandas column (which is one) instead of
            the number of element contained in the Series object (the number of features). While I can add a for loop
            in to get the number inside the list, it is not very clean.
    """
    def __init__(self, df_from_attributes, df_from_predictions, df_from_data):
        self.data = df_from_data.drop(['Unnamed: 0'], axis=1)
        df_from_attributes = df_from_attributes.rename(columns=lambda x: re.sub("predictions_stats.", "", x))
        self.params = df_from_attributes.filter(regex="params.", axis=1).rename(columns=lambda x: re.sub("params.", "", x))
        self.predictions_stats = df_from_attributes.loc[:, 'r2_raw':'time_std']
        self.run_name = df_from_attributes['run_name'].values[0]
        self.algorithm = df_from_attributes['algorithm'].values[0]
        self.feat_method_name = df_from_attributes['feat_method_name'].values[0]
        self.tuned = str(df_from_attributes['tuned'].values[0])
        self.n_test = int(df_from_attributes['n_test'])
        self.feat_time = float(df_from_attributes['feat_time'])
        self.date = df_from_attributes['date'].values[0]
        self.test_time = float(df_from_attributes["time_avg"])
        self.feature_length = int(df_from_attributes['feature_length'].values[0])
        self.feature_list = df_from_attributes['feature_list'].values[0]
        self.n_train = int(df_from_attributes['n_train'].values[0])
        self.dataset = df_from_attributes['dataset'].values[0]
        self.target_name = df_from_attributes['target_name'].values[0]
        self.train_percent = float(df_from_attributes['train_percent'].values[0])
        self.test_percent = float(df_from_attributes['test_percent'].values[0])
        self.val_percent = float(df_from_attributes['val_percent'].values[0])
        self.random_seed = int(df_from_attributes['random_seed'].values[0])
        self.params_list = list(df_from_attributes.filter(regex='params.'))
        self.canonical_smiles = fragments.canonical_smiles(list(df_from_data['smiles']))
        self.target_array = list(df_from_data['target'])
        self.n_val = int(df_from_attributes['n_val'])
        self.predictions = df_from_predictions
        pva = self.predictions
        self.r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
        self.mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
        self.rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
        self.predicted = list(pva['pred_avg'])  # List of predicted value for test molecules
        self.df_smiles = df_from_data.iloc[:, [1, 2]]
        self.test_mol_dict = pd.DataFrame({'smiles': list(pva['smiles']), 'predicted': list(pva['pred_avg']),
                                           'uncertainty': list(pva['pred_std'])}).to_dict('records')
        self.train_molecules = split_molecules(df_from_data, 'train')
        self.test_molecules = split_molecules(df_from_data, 'test')
        self.val_molecules = split_molecules(df_from_data, 'val')
        self.feat_meth = df_from_attributes['feat_meth'].values[0]
        self.tune_algorithm_name = df_from_attributes['tune_algorithm_name'].values[0]
        if "rdkit2d" in self.feat_method_name:
            self.rdkit2d_features = df_from_data.loc[:, 'smiles':'qed']
        else:
            pass
        if self.tuned:
            self.cv_folds = int(df_from_attributes['cv_folds'].values[0])
            self.tune_time = float(df_from_attributes['tune_time'].values[0])
            self.cp_delta = float(df_from_attributes['cp_delta'].values[0])
            self.cp_n_best = int(df_from_attributes['cp_n_best'].values[0])
            self.opt_iter = int(df_from_attributes['opt_iter'].values[0])
        else:
            self.cv_folds = 0
            self.tune_time = 0
            self.cp_delta = 0
            self.cp_n_best = 0
            self.opt_iter = 0

    def to_neo4j(self):
        """"""
        nodes_to_neo4j.nodes(self)
        rel_to_neo4j.relationships(self)