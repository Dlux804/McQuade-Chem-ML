"""
Objective: This file contains different dictionaries that are used throughout the pipeline
"""


def target_name_grid(dataset):
    """
    Objective: Return true experimental target for a dataset
    :param dataset:
    :return:
    """
    grid = {
            'ESOL.csv': 'water_sol',
            'lipo_raw.csv': 'logP',
            'water-energy.csv': 'hydration_energy',
            'logP14k.csv': 'logP_kow',
            'jak2_pic50.csv': 'pIC50',
            'Lipo-short.csv': 'logP'
        }

    if dataset in grid:
        return grid[dataset]

    return None


def get_param(algorithm):
    # Dictionary of algorithms and their parameters of interest
    param_dict = {
        "gdb": ['learning_rate', 'max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'n_estimators'],
        "rf": ['max_depth', "max_features", 'min_samples_leaf', 'min_samples_split', 'n_estimators', 'bootstrap'],
        "knn": ["algorithm", 'leaf_size', 'n_neighbors', 'p', "weights"],
        "ada": ['base_estimator', 'learning_rate', 'n_estimators'],
        "svm": ['kernel', 'C', 'gamma', 'epsilon', 'degree'],
        "mlp": ['activation', 'solver', 'alpha', 'learning_rate'],
        "nn": ['n_hidden', 'n_neuron', 'learning_rate', 'drop'],
        "cnn": ['n_hidden', 'n_neuron', 'learning_rate', 'drop']
    }
    return param_dict[algorithm]


def nn_default_param():
    """
    Objective: Return NN default parameters defined in build_nn function located in regressors.py
    :return:
    """
    return {'n_hidden': 2, 'n_neuron': 50, 'learning_rate': 1e-3, 'in_shape': 200, 'drop': 0.1}
