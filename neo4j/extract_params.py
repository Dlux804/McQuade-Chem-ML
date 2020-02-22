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
    params_dict = {
        "gdb": col_gdb,
        "rf": col_rf,
        "knn": col_knn,
        "ada": col_ada
    }
    return params_dict[algo]


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
    :param algorithm:
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


def param_lst(csv, algo):
    """

    :param csv:
    :param algo:
    :return:
    """
    df, param_lst = get_param(csv, algo)
    df_results = df[df.algorithm == algo]
    runs_idx = df_results["algorithm"]
    param_df = pd.DataFrame.from_records(param_lst, columns=param_dict(algo))
    unique_df = param_df.loc[:, ~(param_df == param_df.iloc[0]).all()]
    # final_df["algorithm"] = runs_idx.tolist()
    runs_lst = runs_idx.tolist()
    final_df = unique_df.assign(algorithm=runs_lst)
    print(final_df)
    main_lst = []
    for col in final_df.columns.tolist():
        col_lst = []
        for i in final_df[col]:
            t = re.sub('.*=', '', i)
            col_lst.append(t)
        main_lst.append(col_lst)
    return final_df, main_lst


def param_finaldf(csv, algo):
    """

    :param csv:
    :param algo:
    :return:
    """
    df, main_lst = param_lst(csv, algo)
    # print(runs_idx)
    rotate_lst = list(rotated(main_lst))
    array_rotate = np.array(rotate_lst)
    final_df = pd.DataFrame.from_records(array_rotate, columns=df.columns.tolist())
    # print("Parameter Dataframe\n")
    # print(final_df)
    # final_df.to_csv(algo + "_params.csv")
    return final_df

# param_lst('ml_results2.csv', "rf")
# param_finaldf('ml_results2.csv', "rf")
# algo = ['rf', 'gdb', 'knn', 'ada']
# for i in algo:
#     param_df('ml_results2.csv', i)
