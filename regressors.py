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
from sklearn.model_selection import RandomizedSearchCV
from time import time


def regressor(model, tune=False):
    """Returns model specific regressor function."""

    # Create Dictionary of regressors to be called with self.algorithm as key.
    regressors = {
        'ada': AdaBoostRegressor,
        'rf': RandomForestRegressor,
        'svr': SVR,
        'gdb': GradientBoostingRegressor,
        'mlp': MLPRegressor,
        'knn': KNeighborsRegressor
    }
    return regressors[model]


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
def hyperTune(model, train_features, train_target, grid, folds, iters, jobs=-1): # WHAT is expt? WHY use it?
    # TODO: Add doc string
    print("Starting Hyperparameter tuning\n")
    start_tune = time()
    search_random = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=iters, cv=folds, verbose=2,
                                       random_state=42, n_jobs=jobs)  # TODO change criteria to MSE instead of R2
    # TODO: Set up logging of parameters
    # log.at[exp, 'Parameter Grid'] = grid

    # Fit the random search model
    search_random.fit(train_features, train_target)
    tuned = search_random.best_params_

    # Calculate time to tune parameters
    stop_tune = time()
    tune_time = stop_tune - start_tune
    print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
    print(tuned)
    return tuned, tune_time






