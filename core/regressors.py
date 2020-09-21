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
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Conv2D
from tensorflow.keras.metrics import RootMeanSquaredError
from tqdm import tqdm
from core.storage.misc import __cv_results__, __fix_ada_dictionary__
# monkey patch to fix skopt and sklearn.  Requires downgrade to sklearn 0.23
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

import skopt
from skopt import BayesSearchCV
from skopt import callbacks

# end monkey patch

# TODO: Add other tuning algorithms and create a variable that stores the algorithm's name


def build_nn(n_hidden=2, n_neuron=50, learning_rate=1e-3, in_shape=200, drop=0.1):
    """
    Create neural network architecture and compile.  Accepts number of hiiden layers, number of neurons,
    learning rate, and input shape. Returns compiled model.

    Keyword Arguments:
        n_hidden (integer): Number of hidden layers added to model, excludes input and output layer. Default = 2
        n_neuron (integer): Number of neurons to add to each hidden layer. Default = 50
        learning_rate (float):  Model learning rate that is passed to model optimizer.
                                Smaller values are slower, High values are prone to unstable training. Default = 0.001
        in_shape (integer): Input dimension should match number of features.  Default = 200 but should be overridden.
        drop (float): Dropout probability.  1 means drop everything, 0 means drop nothing. Default = 0.
                        Recommended = 0.2-0.6
    """

    model = keras.models.Sequential()
    # use dropout layer as input.
    model.add(keras.layers.Dropout(drop, input_shape=(in_shape,)))  # in_shape should be iterable (tuple)
    # model.add(keras.layers.InputLayer(input_shape=in_shape))  # input layer.  How to handle shape?
    for layer in range(n_hidden):  # create hidden layers
        model.add(keras.layers.Dense(n_neuron, activation="relu"))
        model.add(keras.layers.Dropout(drop))  # add dropout to model after the a dense layer

    model.add(keras.layers.Dense(1))  # output layer
    # TODO Add optimizer selection as keyword arg for tuning
    # optimizer = keras.optimizers.SGD(lr=learning_rate)  # this is a point to vary.  Dict could help call other ones.
    # optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=[RootMeanSquaredError(name='rmse')])

    return model


def build_cnn(n_hidden=2, n_neuron=50, learning_rate=1e-3, in_shape=200, drop=0.1):
    """
    Objective: Create Convolutional Neural Network Architecture for regression
    How to get started: https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html
    How to tune: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    Valuable info:
    https://github.com/keras-team/keras/blob/8a8ef43ffcf8d95d2880da073bbcee73e51aad48/docs/templates/getting-started/sequential-model-guide.md
    :param input_shape:
    :return:
    """
    model = keras.models.Sequential()
    # Experiment on both Conv1D and Conv2D is needed.
    # According to most tutorial, the input_shape depends on the shape of your data. In this case, the shape of our data
    # is (number_of_features, 1) since
    model.add(keras.layers.Dropout(drop, input_shape=(in_shape, 1)))  # in_shape should be iterable (tuple)
    # model.add(keras.layers.InputLayer(input_shape=in_shape))  # input layer.  How to handle shape?
    for layer in range(n_hidden):  # create hidden layers
        model.add(keras.layers.Dense(n_neuron, activation="relu"))
        model.add(keras.layers.Dropout(drop))  # add dropout to model after the a dense layer
    model.add(Conv1D(32, 2, activation='relu', input_shape=(in_shape, 1)))

    # model.add(keras.layers.MaxPooling1D(pool_size=3))
    # model.add(Conv1D(64, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=[RootMeanSquaredError(name='rmse')])
    return model


def wrapKeras(self, build_func):
    """
    Wraps up a Keras model to appear as sklearn Regressor for use in hyper parameter tuning.
    :param self: For use in MLmodel class instance.
    :param build_func: Callable function that builds Keras model.
    :return: Regressor() like function for use with sklearn based optimization.
    """

    # pass non-hyper params here
    # if model has been tuned, it should have 'parrams' attribute
    # create regressor instance with tuned parameters
    # if self.algorithm == 'nn':
    if hasattr(self, 'params'):
            self.estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape,
                                                                        **self.params)

    # has not been tuned and no params have been supplied, so use default.
    else:
            self.estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape)
    #
    # if self.algorithm == 'cnn':
    #     if hasattr(self, 'params'):
    #         self.estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, **self.params)
    #
    #     # has not been tuned and no params have been supplied, so use default.
    #     else:
    #         self.estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func)

def get_regressor(self, call=False):
    """
    Returns model specific regressor function.
    Optional argument to create callable or instantiated instance.
    """

    # Create Dictionary of regressors to be called with self.algorithm as key.
    skl_regs = {
        'ada': AdaBoostRegressor,
        'rf': RandomForestRegressor,
        'svm': SVR,
        'gdb': GradientBoostingRegressor,
        'mlp': MLPRegressor,
        'knn': KNeighborsRegressor
    }
    if self.algorithm in skl_regs.keys():
        self.task_type = 'regression'

        # if want callable regressor
        if call:
            self.estimator = skl_regs[self.algorithm]

        # return instance with either default or tuned params
        else:
            if hasattr(self, 'params'):  # has been tuned
                self.estimator = skl_regs[self.algorithm](**self.params)
            elif self.algorithm in ['gdb', 'rf']:
                self.estimator = skl_regs[self.algorithm](ccp_alpha=0.1)
            else:  # use default params
                self.estimator = skl_regs[self.algorithm]()

    if self.algorithm in ['nn', 'cnn']:  # neural network

        # set a checkpoint file to save the model
        chkpt_cb = keras.callbacks.ModelCheckpoint(self.run_name + '.h5', save_best_only=True)
        # set up early stopping callback to avoid wasted resources
        stop_cb = keras.callbacks.EarlyStopping(patience=10,  # number of epochs to wait for progress
                                                restore_best_weights=True)

        # params to pass to Keras fit method that don't match sklearn params
        self.fit_params = {'epochs': 100,
                           'batch_size': 50,
                           'callbacks': [chkpt_cb, stop_cb],
                           'validation_data': (self.val_features, self.val_target)
                           }
        # wrap regressor like a sklearn regressor
        if self.algorithm == 'nn':
            wrapKeras(self, build_func=build_nn)
        else:
            wrapKeras(self, build_func=build_cnn)


# for making a progress bar for skopt
class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


# def hyperTune(model, train_features, train_target, grid, folds, iters, jobs=-1, epochs = 50):
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
    if self.algorithm in ['nn', 'cnn']:
        n_jobs = 1  # nn cannot run hyper tuning in parallel while using GPU.
    else:
        self.fit_params = None  # sklearn models don't need fit params

    # set up Bayes Search
    bayes = BayesSearchCV(
        estimator=self.estimator,  # what regressor to use
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
    # TODO try different scaling with delta
    # self.cp_delta = 0.05
    self.cp_delta = (0.08 - min(self.train_target.min()))/(max(self.train_target.max()) - min(self.train_target.min()))  # Min max scaling
    print("cp_delta is : ", self.cp_delta)
    # self.cp_delta = delta_std * (self.train_target.max() - self.train_target.min()) + self.train_target.min()
    self.cp_n_best = 5

    """ 
    Every optimization model in skopt saved all their scores in a built-in lists.
    When called, DeltaYStopper will access this list and sort this list from lowest number to highest number.
    It then take the difference between the number in the n_best position and the first number and
    compares it to delta. If the difference is smaller or equal to delta, the optimization will be stopped.
    """

    # print("delta and n_best is {0} and {1}".format(self.cp_delta, self.cp_n_best))
    deltay = callbacks.DeltaYStopper(self.cp_delta, self.cp_n_best)

    # Fit the Bayes search model, use early stopping
    # if self.algorithm in ['nn', 'cnn']:
    bayes.fit(self.train_features,
                  self.train_target,
                  callback=[tqdm_skopt(total=self.opt_iter, position=0, desc="Bayesian Parameter Optimization"),
                        checkpoint_saver, deltay])
    # else:
    # bayes.fit(self.train_features,
    #               self.train_target,
    #               callback=[tqdm_skopt(total=self.opt_iter, position=0, desc="Bayesian Parameter Optimization"),
    #                     checkpoint_saver, deltay]
    #                 )
    # else:  # nn no early stopping


    # collect best parameters from tuning
    self.params = bayes.best_params_
    tune_score = bayes.best_score_

    self.cv_results = __cv_results__(bayes.cv_results_)
    if self.algorithm == 'ada':
        self.cv_results = __fix_ada_dictionary__(self.cv_results)

    # update the regressor with best parameters
    self.get_regressor(call=False)

    # Calculate time to tune parameters
    stop_tune = time()
    self.tune_time = stop_tune - start_tune
    print('Best Parameter Found After ', (stop_tune - start_tune), "sec\n")
    print('Best params achieve a test score of', tune_score, ':')
    print('Model hyper paramters are:', self.params)


