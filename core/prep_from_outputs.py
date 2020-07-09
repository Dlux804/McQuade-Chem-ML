"""
Objective: This script stores all data from output files. The data will then be used to create nodes and
relationships based on our ontology to Neo4j
"""
from core import fragments
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# TODO: Add documentation


def split_molecules(dataframe, split):
    """

    :param dataframe:
    :param split:
    :return:
    """
    split = ''.join([split[0].lower(), split[1:]])
    df = dataframe.loc[dataframe['in_set'] == split]
    df = df.reset_index(drop=True)
    df = df.drop(['Unnamed: 0'], axis=1)
    molecules = fragments.canonical_smiles(list(df['smiles']))
    return molecules


class Prep:
    """
    Class containing all the data needed for creating nodes and relationships from output files
    """
    def __init__(self, df_from_attributes, df_from_predictions, df_from_data):
        self.df_from_data = df_from_data
        self.df_from_attributes = df_from_attributes
        self.run_name = df_from_attributes['run_name'].to_string(index=False)
        self.algorithm = df_from_attributes['algorithm'].to_string(index=False)
        self.feat_method_name = df_from_attributes['feat_method_name'].tolist()
        self.tuned = df_from_attributes['tuned'].to_string(index=False)
        self.cv_folds = int(df_from_attributes['cv_folds'])
        self.n_test = int(df_from_attributes['n_test'])
        self.tune_time = float(df_from_attributes['tune_time'])
        self.cp_delta = int(df_from_attributes['cp_delta'])
        self.cp_n_best = int(df_from_attributes['cp_n_best'])
        self.opt_iter = int(df_from_attributes['opt_iter'])
        self.feat_time = float(df_from_attributes['feat_time'])
        self.date = df_from_attributes['date'].to_string(float_format=True, index=False)
        self.tune_time = float(df_from_attributes['tune_time'])
        self.test_time = float(df_from_attributes["predictions_stats.time_avg"])
        self.num_feature_list = len(list(df_from_attributes['feature_list']))
        self.n_train = int(df_from_attributes['n_train'])
        self.dataset_str = df_from_attributes['dataset'].to_string(float_format=True, index=False)
        self.target_name = df_from_attributes['target_name'].to_string(float_format=True, index=False)
        self.test_percent = float(df_from_attributes['test_percent'])
        self.train_percent = float(df_from_attributes['train_percent'])
        self.random_seed = int(df_from_attributes['random_seed'])
        self.val_percent = float(df_from_attributes['val_percent'])
        self.params_list = list(df_from_attributes.filter(regex='params.'))
        self.canonical_smiles = fragments.canonical_smiles(list(df_from_data['smiles']))
        self.target_array = list(df_from_data['target'])
        self.n_val = int(df_from_attributes['n_val'])
        self.rdkit2d_features = df_from_data.loc[:, 'BalabanJ':'qed']
        self.features_col = list(self.rdkit2d_features.columns)
        pva = df_from_predictions
        self.r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
        self.mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
        self.rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
        self.predicted = list(pva['pred_avg'])  # List of predicted value for test molecules
