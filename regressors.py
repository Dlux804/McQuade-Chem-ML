'''
List of the different regressors associated with each learning algorithm.
Employ the function dictionary to call regressor functions by model keyword.
'''

from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def runRegress(model, model_name,train_features, test_features, train_target, test_target, expt):
    # TODO: Add doc string
    start_reg = time()
    print("Starting " + model_name + expt +'\n')
    pva, time_var = defaultML(model,model_name, train_features, test_features, train_target, test_target, expt)
    r2_dict, mse_dict, rmse_dict = pva_graphs(pva,model_name, expt)
    stop_reg = time()
    print(model_name + expt+ ' Finished after ',(stop_reg-start_reg),"sec\n")
    return r2_dict, mse_dict, rmse_dict, time_var


# Hyper parameter tunes using RandomizedSearchCV
def hyperTune(model, model_name, train_features, train_target, grid, folds, iters, expt): # WHAT is expt? WHY use it?
    # TODO: Add doc string
    print("Starting Hyperparameter tuning\n")
    start_tune = time()
    search_random = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=iters, cv=folds, verbose=2,
                                       random_state=42, n_jobs=4)
    log.at[exp, 'Parameter Grid'] = grid
    # Fit the random search model
    search_random.fit(train_features, train_target)
    tuned = search_random.best_params_
    param_dict = {str(model_name) + '-' + expt: tuned}

    stop_tune = time()
    time_par = {str(model_name) + '-' + expt: (stop_tune - start_tune)}
    print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
    print(tuned)
    return tuned, param_dict, time_par


def regressor(model, tune=False):
    """
    Run machine learning regression specific to learning algorithm.
    Returns dictionaries for r2, mse, rmse, and hyperparameters. Returns timing values as well.

    Keyword Arguments:
        tune -- Do Hyperparameter tuning (default=False)
    """

    # Create Dictionary of regressors to be called with self.algorithm as key.
    regressors = {
        'ada': AdaBoostRegressor,
        'rf': RandomForestRegressor,
        'svr': SVR,
        'gdb': GradientBoostingRegressor,
        'mlp': MLPRegressor,
        'knn': KNeighborsRegressor
    }
    return regressors[model]()



