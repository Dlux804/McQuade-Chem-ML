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
from skopt import BayesSearchCV
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
    # search_random = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=iters, cv=folds, verbose=2,

    # set up Bayes Search
    bayes = BayesSearchCV(
        estimator=model,  # what regressor to use
        search_spaces=grid,  # hyper parameters to search through
        n_iter=iters,  # number of combos tried
        random_state=42,  # random seed
        verbose=3,  # output print level
        scoring='neg_mean_squared_error',  # scoring function to use (RMSE)
        n_jobs=30,  # number of parallel jobs
        cv=folds  # number of cross-val folds to use
    )

    # Fit the random search model
    # search_random.fit(train_features, train_target)
    # tuned = search_random.best_params_

    # Fit the Bayes search model
    bayes.fit(train_features, train_target)
    tuned = bayes.best_params_
    tune_score = bayes.best_score_

    # Calculate time to tune parameters
    stop_tune = time()
    tune_time = stop_tune - start_tune
    print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
    print('Best params achieve a test score of', tune_score, ':')
    print(tuned)
    return tuned, tune_time






