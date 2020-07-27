"""
Get
"""

"""
List of the different regressors associated with each learning algorithm.
Employ the function dictionary to call regressor functions by model keyword.
"""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from time import time
from exp_core.neural_network import wrapKeras
from tqdm import tqdm
from tensorflow import keras
# monkey patch to fix skopt and sklearn.  Requires downgrade to sklearn 0.23
from numpy.ma import MaskedArray
import sklearn.utils.fixes
#
sklearn.utils.fixes.MaskedArray = MaskedArray

import skopt
from skopt import BayesSearchCV
from skopt import callbacks

# end monkey patch
from exp_core.name import Name
from exp_core.split import Split
from exp_core.grid import Grid
# TODO: Add other tuning algorithms and create a variable that stores the algorithm's name


# for making a progress bar for skopt
class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


class Estimator(Name, Split, Grid):
    def __init__(self, algorithm, dataset, target, feat_meth, tune, cv_folds, opt_iter):
        Name.__init__(self, algorithm, dataset, target, feat_meth, tune)
        Split.__init__(self, dataset, target)
        Grid.__init__(self, algorithm)
        self.cv_folds = cv_folds
        self.opt_iter = opt_iter

    # @classmethod
    def get_regressor(self, call=False):
        """
        Returns model specific regressor function.
        Optional argument to create callable or instantiated instance.
        """

        # Create Dictionary of regressors to be called with self.algorithm as key.
        skl_regs = {
            'ada': AdaBoostRegressor,
            'rf': RandomForestRegressor,
            'svr': SVR,
            'gdb': GradientBoostingRegressor,
            'mlp': MLPRegressor,
            'knn': KNeighborsRegressor
        }
        if self.algorithm in skl_regs.keys():
            self.task_type = 'regression'

            # if want callable regressor
            if call:
                self.regressor = skl_regs[self.algorithm]

            # return instance with either default or tuned params
            else:
                if hasattr(self, 'params'):  # has been tuned
                    self.regressor = skl_regs[self.algorithm](**self.params)
                else:  # use default params
                    self.regressor = skl_regs[self.algorithm]()

        if self.algorithm == 'nn':  # neural network

            # set a checkpoint file to save the model
            chkpt_cb = keras.callbacks.ModelCheckpoint(self.run_name + '.h5', save_best_only=True)
            # set up early stopping callback to avoid wasted resources
            stop_cb = keras.callbacks.EarlyStopping(patience=10,  # number of epochs to wait for progress
                                                    restore_best_weights=True)

            # params to pass to Keras fit method that don't match sklearn params
            self.fit_params = {'epochs': 100,
                               'callbacks': [chkpt_cb, stop_cb],
                               'validation_data': (self.val_features, self.val_target)
                               }
            # wrap regressor like a sklearn regressor
            wrapKeras(self)


    def hyperTune(self, epochs=50, n_jobs=6):
        """
        Tunes hyper parameters of specified model.

        Inputs:
        algorithm, training features, training targets, hyper-param grid,
        number of cross validation folds to use, number of optimization iterations,

       Keyword arguments
       jobs: number of parallel processes to run.  (Default = -1 --> use all available cores)
       NOTE: jobs has been depreciated since max processes in parallel for Bayes is the number of CV folds

       'neg_mean_squared_error',  # scoring function to use (RMSE)
        """
        print("Starting Hyperparameter tuning\n")
        start_tune = time()
        if self.algorithm == "nn":
            n_jobs = 1  # nn cannot run hyper tuning in parallel while using GPU.
        else:
            self.fit_params = None  # sklearn models don't need fit params

        # set up Bayes Search
        bayes = BayesSearchCV(
            estimator=self.regressor,  # what regressor to use
            search_spaces=self.param_grid,  # hyper parameters to search through
            fit_params=self.fit_params,
            n_iter=self.opt_iter,  # number of combos tried
            random_state=42,  # random seed
            verbose=3,  # output print level
            scoring='neg_mean_squared_error',  # scoring function to use (RMSE)  #TODO needs update for Classification
            n_jobs=n_jobs,  # number of parallel jobs (max = folds)
            cv=self.cv_folds  # number of cross-val folds to use
        )
        self.tune_algorithm_name = str(type(bayes).__name__)
        # if self.algorithm != 'nn':  # non keras model
        checkpoint_saver = callbacks.CheckpointSaver(''.join('./%s_checkpoint.pkl' % self.run_name), compress=9)
        # checkpoint_saver = callbacks.CheckpointSaver(self.run_name + '-check')
        self.cp_delta = 0.05  # TODO delta should be dynamic to scale with target value
        self.cp_n_best = 5

        """ 
        Every optimization model in skopt saved all their scores in a built-in list.
        When called, DeltaYStopper will access this list and sort this list from lowest number to highest number.
        It then take the difference between the number in the n_best position and the first number and
        compares it to delta. If the difference is smaller or equal to delta, the optimization will be stopped.
        """

        # print("delta and n_best is {0} and {1}".format(self.cp_delta, self.cp_n_best))
        deltay = callbacks.DeltaYStopper(delta=self.cp_delta, n_best=self.cp_n_best)

        # Fit the Bayes search model, use early stopping
        bayes.fit(self.train_features,
                  self.train_target,
                  callback=[tqdm_skopt(total=self.opt_iter, position=0, desc="Bayesian Parameter Optimization"),
                            checkpoint_saver,
                            deltay]
                  )
        # else:  # nn no early stopping
        #     bayes.fit(self.train_features,
        #               self.train_target,
        #               callback=[tqdm_skopt(total=self.opt_iter, position=0, desc="Bayesian Parameter Optimization")])

        # collect best parameters from tuning
        self.params = bayes.best_params_
        tune_score = bayes.best_score_

        # update the regressor with best parameters
        self.get_regressor(call=False)

        # Calculate time to tune parameters
        stop_tune = time()
        self.tune_time = stop_tune - start_tune
        print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
        print('Best params achieve a test score of', tune_score, ':')
        print('Model hyper paramters are:', self.params)




