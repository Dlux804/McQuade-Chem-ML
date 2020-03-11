import pandas as pd
import re
import numpy as np


col_gdb = ['alpha', 'ccp_alpha', 'criterion', 'init', 'learning_rate', 'loss', 'max_depth', 'max_features',
           'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split',
           'min_weight_fraction_leaf', 'n_estimators', 'n_iter_no_change', 'presort', 'random_state', 'subsample',
           'tol', 'validation_fraction', 'verbose', 'warm_start']

col_rf = ['bootstrap', 'ccp_alpha', "criterion", 'max_depth', "max_features",
          'max_leaf_nodes', 'max_samples', 'min_impurity_decrease', 'min_impurity_split',
          'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators',
          'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start']

col_knn = ["algorithm", 'leaf_size', "metric", 'metric_params', 'n_jobs', 'n_neighbors', 'p', "weights"]

col_ada = ['base_estimator', 'learning_rate', "loss", 'n_estimators', 'random_state']


def param_dict(algor):
    params_dict = {
        "gdb": col_gdb,
        "rf": col_rf,
        "knn": col_knn,
        "ada": col_ada
    }
    return params_dict[algor]


def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


class Params:
    @staticmethod
    def clean_param(csv):
        """
        Extract parameters and put them into a list
        :param csv:
        :return:
        """
        pre_df = pd.read_csv(csv, index_col=0)
        df_clean = pre_df.dropna()
        df = df_clean.reset_index(drop=True)
        # df_algor = df[df.algorithm == algor]
        dct = df.to_dict('records')
        param_list = []
        param_clean = []
        for i in range(len(dct)):
            row_dict = dct[i]
            params = row_dict['regressor']
            # print(params)
            reg = params[params.find("(") + 1:params.find(")")]
            new_reg = " ".join(reg.split())
            final_reg = new_reg.replace("'", "")
            element = final_reg.split(",")
            param_list.append(element)
            param_clean.append(final_reg)
        drop_df = df.drop(columns='regressor')
        final_df = drop_df.assign(regressor=param_clean)
        return final_df, param_list

    @staticmethod
    def param_lst(csv, algor):
        """

        :param csv:
        :param algor
        :return:
        """
        pre_df, param_list = Params.clean_param(csv)
        df = pre_df.assign(param_lst=param_list)
        df_algor = df[df.algorithm == algor]
        param_list = df_algor["param_lst"].tolist()
        algor_lst = df_algor["algorithm"].tolist()
        regress_lst = df_algor['regressor'].tolist()
        param_df = pd.DataFrame.from_records(param_list, columns=param_dict(algor))
        unique_df = param_df.loc[:, ~(param_df == param_df.iloc[0]).all()]
        # print(unique_df)
        final_df = unique_df.assign(algorithm=algor_lst, regressor=regress_lst)
        # print(df_with_algor)
        col_list = final_df.columns.tolist()
        main_list = []
        for col in col_list:
            col_lst = []
            for i in final_df[col]:
                t = re.sub('.*=', '', i)
                col_lst.append(t)
            main_list.append(col_lst)
        return col_list, main_list

    def param_df(self, csv, algor):
        """

        :param csv:
        :param algor:
        :return:
        """
        col_list, main_list = Params.param_lst(csv, algor)
        rotate_lst = list(rotated(main_list))
        array_rotate = np.array(rotate_lst)
        df_records = pd.DataFrame.from_records(array_rotate, columns=col_list)
        # final_df.to_csv("test.csv")
        return df_records

# params = Params()
# params.param_extract('ml_results3.csv')
# params.param_lst('ml_results3.csv', "gdb")
# params.param_df('ml_results3.csv', "gdb")

