"""
Objective: This script stores all data from output files. The data will then be used to create nodes and
relationships based on our ontology to Neo4j
"""
from core import fragments
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def remove_whitespace(string):
    """
    Objective: Remove whitespaces from string variables
    Intent: Because I gather data from output files based on column names (the data is contained in dataframes
            at this point), they are Pandas Series object. When I convert them into string and remove the index label
            using "to_string(index=False)", there are whitespaces in the string variable. Therefore, I needed to create
            this function to get rid of the whitespaces. I should have found a more clever way to obtain data from
            dataframes instead of doing all this.
    :param string: A string. What did you expect it to be???
    :return:
    """
    return "".join(string.split())


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
        self.df_from_data = df_from_data
        self.df_from_attributes = df_from_attributes
        self.run_name = remove_whitespace(df_from_attributes['run_name'].to_string(index=False))
        self.algorithm = remove_whitespace(df_from_attributes['algorithm'].to_string(index=False))
        self.feat_method_name = df_from_attributes['feat_method_name'].tolist()
        self.tuned = remove_whitespace(df_from_attributes['tuned'].to_string(index=False))
        self.cv_folds = int(df_from_attributes['cv_folds'])
        self.n_test = int(df_from_attributes['n_test'])
        self.tune_time = float(df_from_attributes['tune_time'])
        self.cp_delta = int(df_from_attributes['cp_delta'])
        self.cp_n_best = int(df_from_attributes['cp_n_best'])
        self.opt_iter = int(df_from_attributes['opt_iter'])
        self.feat_time = float(df_from_attributes['feat_time'])
        self.date = remove_whitespace(df_from_attributes['date'].to_string(float_format=True, index=False))
        self.tune_time = float(df_from_attributes['tune_time'])
        self.test_time = float(df_from_attributes["predictions_stats.time_avg"])
        self.num_feature_list = [len(i) for i in list(df_from_attributes['feature_list'])]  # number of features
        self.n_train = int(df_from_attributes['n_train'])
        self.dataset_str = remove_whitespace(df_from_attributes['dataset'].to_string(float_format=True, index=False))
        self.target_name = remove_whitespace(df_from_attributes['target_name'].to_string(float_format=True, index=False))
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
