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
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError
from tqdm import tqdm

# monkey patch to fix skopt and sklearn.  Requires downgrade to sklearn 0.23
from numpy.ma import MaskedArray
import sklearn.utils.fixes
#
sklearn.utils.fixes.MaskedArray = MaskedArray

import skopt
from skopt import BayesSearchCV
from skopt import callbacks

# end monkey patch

def build_nn( n_hidden = 2, n_neuron = 50, learning_rate = 1e-3, in_shape=200, drop=0.1):
    """
    Create neural network architecture and compile.  Accepts number of hiiden layers, number of neurons,
    learning rate, and input shape. Returns compiled model.

    Keyword Arguments:
        n_hidden (integer): Number of hidden layers with n_neurons to be added to model, excludes input and output layer. Default = 2
        n_neuron (integer): Number of neurons to add to each hidden layer. Default = 50
        learning_rate (float):  Model learning rate that is passed to model optimizer.  Smaller values are slower, High values
                        are prone to unstable training. Default = 0.001
        in_shape (integer): Input dimension should match number of features.  Default = 200 but should be overridden.
        drop (float): Dropout probability.  1 means drop everything, 0 means drop nothing. Default = 0.
                        Recommended = 0.2-0.6
    """

    model = keras.models.Sequential()
    # model.add(keras.layers.Dropout(drop, input_shape=self.in_shape))  # use dropout layer as input.
    model.add(keras.layers.InputLayer(input_shape=in_shape))  # input layer.  How to handle shape?
    for layer in range(n_hidden):  # create hidden layers
        model.add(keras.layers.Dense(n_neuron, activation="relu"))
        model.add(keras.layers.Dropout(drop))  # add dropout to model after the a dense layer

    model.add(keras.layers.Dense(1))  # output layer
    # TODO Add optimizer selection as keyword arg
    # optimizer = keras.optimizers.SGD(lr=learning_rate)  # this is a point to vary.  Dict could help call other ones.
    # optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=[RootMeanSquaredError(name='rmse')])

    return model


def wrapKeras(self, build_func=build_nn):
    """
    Wraps up a Keras model to appear as sklearn Regressor for use in hyper parameter tuning.
    :param build_func: Callable function that builds Keras model.
    :param in_shape: Input dimension.  Must match number of features.
    :return: Regressor() like function for use with sklearn based optimization.
    """

    # pass non-hyper params here
    if hasattr(self, 'params'):
        self.regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape,
                                                                    **self.params)
    else:
        self.regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape)


def get_regressor(self, call=False):
    """Returns model specific regressor function."""

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
        if call:
            self.regressor = skl_regs[self.algorithm]
        else:
            if hasattr(self, 'params'):
                self.regressor = skl_regs[self.algorithm](**self.params)
            else:
                self.regressor = skl_regs[self.algorithm]()

    if self.algorithm == 'nn':  # neural network
        # compile nn model
        # reg = build_nn(self)
        # wrap it like a sklearn regressor
        # set a checkpoint file to save the model
        chkpt_cb = keras.callbacks.ModelCheckpoint( self.run_name + '.h5', save_best_only=True)
        # set up early stopping callback to avoid wasted resources
        stop_cb = keras.callbacks.EarlyStopping(patience=10,  # number of epochs to wait for progress
                                                restore_best_weights=True)

        self.fit_params = {'epochs': 100,
                           'callbacks': [chkpt_cb, stop_cb],
                           'validation_data': (self.val_features, self.val_target)
                           }
        wrapKeras(self)


# for making a progress bar for skopt
class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


# def hyperTune(model, train_features, train_target, grid, folds, iters, jobs=-1, epochs = 50):
def hyperTune(self, epochs=50,n_jobs=6):
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
        n_jobs = 1
        # # set a checkpoint file to save the model
        # chkpt_cb = keras.callbacks.ModelCheckpoint(self.run_name+'.h5', save_best_only=True)
        # # set up early stopping callback to avoid wasted resources
        # stop_cb = keras.callbacks.EarlyStopping(patience=10,  # number of epochs to wait for progress
        #                                         restore_best_weights=True)
        #
        # self.fit_params = {'epochs': 100,
        #               'callbacks': [chkpt_cb, stop_cb],
        #               'validation_data': (self.val_features,self.val_target)
        #                     }
    else:
        self.fit_params = None

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
    if self.algorithm != 'nn':  # non keras model
        checkpoint_saver = callbacks.CheckpointSaver(''.join('./%s_checkpoint.pkl' % self.run_name), compress=9)
        # checkpoint_saver = callbacks.CheckpointSaver(self.run_name + '-check')
        self.cp_delta = 0.05  # TODO delta should be dynamic to match target value scales.  Score scales with measurement
        self.cp_n_best = 10

        """ Every optimization model in skopt saved all their scores in a built-in list. When called, DeltaYStopper will
        access this list and sort this list from lowest number to highest number. It then take the difference between the
        number in the n_best position and the first number and compare it to delta. If the difference is smaller or equal
        to delta, the optimization will be stopped.
        """

        # print("delta and n_best is {0} and {1}".format(self.cp_delta, self.cp_n_best))
        deltay = callbacks.DeltaYStopper(self.cp_delta, self.cp_n_best)

        # Fit the Bayes search model
        bayes.fit(self.train_features, self.train_target, callback=[tqdm_skopt(total=self.opt_iter,position=0, desc="Bayesian Parameter Optimization"),checkpoint_saver, deltay])
    else:  # nn
        bayes.fit(self.train_features, self.train_target, callback=[tqdm_skopt(total=self.opt_iter, position=0, desc="Bayesian Parameter Optimization")])

    self.params = bayes.best_params_
    tune_score = bayes.best_score_

    # update the regressor with best parameters
    self.get_regressor(call=False)
    # print(self.regressor.get_params)
    # if self.algorithm != 'nn':
    #     # only necessary for sklearn
    #     self.regressor = self.regressor(**self.params)


    # Calculate time to tune parameters
    stop_tune = time()
    self.tune_time = stop_tune - start_tune
    print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
    print('Best params achieve a test score of', tune_score, ':')
    print('Model hyper paramters are:', self.params)


