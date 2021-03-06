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



def hyperTune(model, train_features, train_target, grid, folds, iters, jobs=-1): # WHAT is expt? WHY use it?
    """
    Tunes hyper parameters of specified model.

    Inputs:
    algorithm, training features, training targets, hyper-param grid,
    number of cross validation folds to use, number of optimization iterations,

   Keyword arguments
   jobs: number of parallel processes to run.  (Default = -1 --> use all available cores)
   NOTE: jobs has been depreciated since max processes in parallel for Bayes is the number of CV folds

    """
    print("Starting Hyperparameter tuning\n")
    start_tune = time()

    # set up Bayes Search
    bayes = BayesSearchCV(
        estimator=model,  # what regressor to use
        search_spaces=grid,  # hyper parameters to search through
        n_iter=iters,  # number of combos tried
        random_state=42,  # random seed
        verbose=3,  # output print level
        scoring='neg_mean_squared_error',  # scoring function to use (RMSE)
        n_jobs=folds,  # number of parallel jobs (max = folds)
        cv=folds  # number of cross-val folds to use
    )

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
