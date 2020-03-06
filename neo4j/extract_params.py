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
    def param_extract(csv, algor):
        """
        Extract parameters and put them into a list
        :param csv:
        :param algor
        :return:
        """
        df = pd.read_csv(csv)
        df_clean = df[df.tuneTime != 0]
        df_algor = df_clean[df_clean.algorithm == algor]
        dct = df_algor.to_dict('records')
        param_list = []
        param_clean = []
        for i in range(len(dct)):
            row_dict = dct[i]
            params = row_dict['regressor']
            # print(params)
            reg = params[params.find("(") + 1:params.find(")")]
            new_reg = " ".join(reg.split())
            final_reg = new_reg.replace("'", "")
            # print(final_reg)
            element = final_reg.split(",")
            param_list.append(element)
            param_clean.append(final_reg)
        # print(param_list)
        return df_algor, param_list, param_clean

    @staticmethod
    def param_lst(csv, algor):
        """

        :param csv:
        :param algor
        :return:
        """
        df_algor, param_list, param_clean = Params.param_extract(csv, algor)
        # df_results = df_clean[df_clean.algorithm == algor]
        runs_idx = df_algor["algorithm"]
        param_df = pd.DataFrame.from_records(param_list, columns=param_dict(algor))
        unique_df = param_df.loc[:, ~(param_df == param_df.iloc[0]).all()]
        # print(unique_df)
        runs_lst = runs_idx.tolist()
        df_new_col = unique_df.assign(algorithm=runs_lst)
        # print(df_with_algor)
        col_list = df_new_col.columns.tolist()
        main_list = []
        for col in col_list:
            col_lst = []
            for i in df_new_col[col]:
                if i != None:
                    t = re.sub('.*=', '', i)
                    col_lst.append(t)
                else:
                    continue
            main_list.append(col_lst)
        return col_list, main_list, param_clean

    def param_df(self, csv, algor):
        """

        :param csv:
        :param algor:
        :return:
        """
        col_list, main_list, param_clean = Params.param_lst(csv, algor)
        rotate_lst = list(rotated(main_list))
        array_rotate = np.array(rotate_lst)
        df_records = pd.DataFrame.from_records(array_rotate, columns=col_list)
        final_df = df_records.assign(regressor=param_clean)
        # print("Parameter Dataframe\n")
        # print(final_df)
        # final_df.to_csv("test.csv")
        return final_df

# params = Params()
# params.param_extract('ml_results3.csv', "gdb")
# params.param_lst('ml_results3.csv', "gdb")
# params.param_df('ml_results3.csv', "gdb")

