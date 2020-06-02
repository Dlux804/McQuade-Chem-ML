'''
List of the different regressors associated with each learning algorithm.
Employ the function dictionary to call regressor functions by model keyword.
'''

from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from skopt import BayesSearchCV
from time import time
from tensorflow import keras
# import tensorflow as tf  # is this needed or just Keras?


def build_nn(n_hidden = 2, n_neuron = 50, learning_rate = 1e-3, in_shape=[200], drop=0.0):
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
    model.add(keras.layers.Dropout(drop, input_shape=in_shape))  # use dropout layer as input.
    # model.add(keras.layers.InputLayer(input_shape=in_shape))  # input layer.  How to handle shape?
    for layer in range(n_hidden):  # create hidden layers
        model.add(keras.layers.Dense(n_neuron, activation="relu"))
        model.add(keras.layers.Dropout(drop))  # add dropout to model after the a dense layer

    model.add(keras.layers.Dense(1))  # output layer
    # TODO Add optimizer selection as keyword arg
    # optimizer = keras.optimizers.SGD(lr=learning_rate)  # this is a point to vary.  Dict could help call other ones.
    # optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def wrapKeras(build_func, in_shape):
    """
    Wraps up a Keras model to appear as sklearn Regressor for use in hyper parameter tuning.
    :param build_func: Callable function that builds Keras model.
    :param in_shape: Input dimension.  Must match number of features.
    :return: Regressor() like function for use with sklearn based optimization.
    """
    return keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_nn, in_shape=200)  # pass non-hyper params here


# def hyperTune(model, train_features, train_target, grid, folds, iters, jobs=-1, epochs = 50):
def hyperTune(self, jobs=-1, epochs=50):
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
    if model == "nn":
        fit_params = "callbacks"  # pseudo code.  add callbacks, epochs
        #  {'epochs': 50, 'callbacks': [chkpt_cb, stop_cb], 'validation_data': (val_features,val_target)}
    else:
        fit_params = None

    # set up Bayes Search
    bayes = BayesSearchCV(
        estimator=self.regressor,  # what regressor to use
        search_spaces=self.grid,  # hyper parameters to search through
        fit_params= self.callbacks,
        n_iter=self.opt_iter,  # number of combos tried
        random_state=42,  # random seed
        verbose=3,  # output print level
        scoring='neg_mean_squared_error',  # scoring function to use (RMSE)
        n_jobs=self.cv_folds,  # number of parallel jobs (max = folds)
        cv=self.cv_folds  # number of cross-val folds to use
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


def regressor(self):
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
    if model in regressors.keys():
        return regressors[model]
    else:  # neural network
        pass

    return regressors[model]