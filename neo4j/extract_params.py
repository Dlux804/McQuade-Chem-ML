import pandas as pd
import re
import numpy as np


col_gdb = ['alpha', 'ccp_alpha', 'criterion', 'init', 'learning_rate', 'loss', 'max_depth', 'max_features',
           'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split',
           'min_weight_fraction_leaf', 'n_estimators', 'n_iter_no_change', 'presort', 'random_state', 'subsample',
           'tol', 'validation_fraction', 'verbose', 'warm_start']

col_rf = ['bootstrap', ' ccp_alpha', "criterion", ' max_depth', " max_features",
          'max_leaf_nodes', ' max_samples', ' min_impurity_decrease', ' min_impurity_split',
          'min_samples_leaf', ' min_samples_split', ' min_weight_fraction_leaf', ' n_estimators',
          'n_jobs', ' oob_score', ' random_state', ' verbose', ' warm_start']

col_knn = ["algorithm", ' leaf_size', " metric", ' metric_params', ' n_jobs', ' n_neighbors', ' p', " weights"]

col_ada = ['base_estimator', ' learning_rate', " loss", ' n_estimators', ' random_state']


def param_dict(algo):
    param_dict = {
        "gdb": col_gdb,
        "rf": col_rf,
        "knn": col_knn,
        "ada": col_ada
    }
    return param_dict[algo]

def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


def get_param(file, algorithm):
    """
    Extract parameters and put them into a list
    :param file:
    :param algo:
    :return:
    """
    df = pd.read_csv(file)
    df = df[df.tuneTime != 0]
    dct = df.to_dict('records')
    gdbparam_lst = []
    for i in range(len(dct)):
        row_dict = dct[i]
        if row_dict['algorithm'] == algorithm:
            params = row_dict['regressor']
            # print(params)
            reg = params[params.find("(")+1:params.find(")")]
            new_reg = " ".join(reg.split())
            # print(new_reg)
            element = new_reg.split(",")
            # print(element)
            gdbparam_lst.append(element)
    return df, gdbparam_lst


def param_lst(csv, algorithm):
    """

    :param csv:
    :param algorithm:
    :return:
    """
    df, param_lst = get_param(csv, algorithm)
    df_results = df[df.algorithm == algorithm]
    param_df = pd.DataFrame.from_records(param_lst, index=df_results["Run#"], columns=param_dict(algorithm))
    final_df = param_df.loc[:, ~(param_df == param_df.iloc[0]).all()]
    main_lst = []
    for col in final_df.columns.tolist():
        col_lst = []
        for i in final_df[col]:
            t = re.sub('.*=', '', i)
            col_lst.append(t)
        main_lst.append(col_lst)
    return main_lst

main_lst = get_param('ml_results2.csv', 'gdb')
print(main_lst)

# df_results = df[df.algorithm == "gdb"]
# gdbparam_df = pd.DataFrame.from_records(param_lst, index=df_results["Run#"], columns=col_gdb)
# finalgdb_df = gdbparam_df.loc[:, ~(gdbparam_df == gdbparam_df.iloc[0]).all()]


# main_lst = []
# for col in finalgdb_df.columns.tolist():
#     col_lst = []
#     for i in finalgdb_df[col]:
#         t = re.sub('.*=', '', i)
#         # print(t)
#         col_lst.append(t)
#     # print(col_lst)
#     main_lst.append(col_lst)
#

#
# rotate = list(rotated(main_lst))
#
# new_rotate = np.array(rotate)# print(main_lst)
# finalgdb_df = pd.DataFrame.from_records(new_rotate, index=df_results["Run#"], columns=finalgdb_df.columns.tolist())
#
# finalgdb_df.to_csv("gdb_param.csv")
# print(finalgdb_df)