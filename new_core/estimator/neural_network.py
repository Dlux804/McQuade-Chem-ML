"""
Objective: Prepare Keras' neural network
"""

from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError


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


def wrapKeras(self, build_func=build_nn):
    """
    Wraps up a Keras model to appear as sklearn Regressor for use in hyper parameter tuning.
    :param self: For use in MLmodel class instance.
    :param build_func: Callable function that builds Keras model.
    :return: Regressor() like function for use with sklearn based optimization.
    """

    # pass non-hyper params here
    # if model has been tuned, it should have 'parrams' attribute
    # create regressor instance with tuned parameters
    if hasattr(self, 'params'):
        self.regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape,
                                                                    **self.params)

    # has not been tuned and no params have been supplied, so use default.
    else:
        self.regressor = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_func, in_shape=self.in_shape)