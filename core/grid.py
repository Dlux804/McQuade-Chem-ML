# TODO: ADD DOC STRINGS!!!

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor


def ada_paramgrid():
    """ Defines hyper parameters for adaboost """
    base_estimator = [tree.DecisionTreeRegressor(max_features='sqrt', splitter='best', max_depth=3), GradientBoostingRegressor(), SVR(kernel = 'linear'),RandomForestRegressor(n_estimators=500)]
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 30)]
    learning_rate = [0.001,0.005,0.01,0.05,0.1,0.5,1]

    param_grid = {
        'base_estimator': base_estimator,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }
    return param_grid


def rf_paramgrid():
    """ Defines hyper parameters for random forest """
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)] #Number of trees
    max_features = ['auto', 'sqrt']     # Number of features to consider at every split
    max_depth = [int(x) for x in np.linspace(1, 30, num = 11)] # Maximum number of levels in tree
    min_samples_split = [2, 4 ,6 ,8, 10] # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2,3, 4,5,6]# Minimum number of samples required at each leaf node
    bootstrap = [True, False]# Method of selecting samples for training each tree
    # Create the random grid
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    return param_grid


def svr_paramgrid():
    """ Defines hyper parameters for supoort vector regression """
# Kernel functions
    kernel = ['rbf', 'poly', 'linear']
# Penalty parameter C of the error term.
    Cs = [0.001, 0.005 ,0.01, 0.05 ,0.1, 0.5, 1, 5,10,100]
# epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
    epsilon = [0.1,0.2,0.3,0.4, 0.5,0.6]
#  Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    gammas = [0.001, 0.005 ,0.01, 0.05 ,0.1, 0.5, 1]
# Degree of the polynomial kernel function (‘poly’)
    degrees = [1,2,3,4,5]
    param_grid = {'kernel': kernel,'C': Cs, 'gamma' : gammas, 'epsilon': epsilon, 'degree' : degrees}
    return param_grid


def gdb_paramgrid():
    """ Defines hyper parameters for gradient decent boost """
# Number of trees
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 20)]
# Number of features to consider at every split
    max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 25, num = 24, endpoint=True)]
# Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(2, 30, num = 10, endpoint=True)]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(2, 30, num = 10, endpoint=True)]
#learning rate
    learning_rate = [0.001,0.005,0.01,0.05,0.1,0.5,1]

    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'learning_rate': learning_rate
    }
    return param_grid


def mlp_paramgrid():
    #Number of hidden layers
    hidden_layer_sizes = [(100,), (100,50,100), (50,100,50), (np.random.randint(low = 50, high = 100, size = 10))]
    #Activation function for the hidden layer.
    activation = ['tanh', 'logistic']
    #The solver for weight optimization.
    solver = ['adam', 'sgd']
    #L2 penalty (regularization term) parameter.
    alpha = [0.0001, 0.0005,0.001,0.005,0.01,0.05,0.1]
    #Learning rate
    learning_rate = ['constant','adaptive', 'invscaling']
    param_grid= {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            }
    return param_grid


def knn_paramgrid():
    #Number of neighbors to use
    n_neighbors = [int(x) for x in np.linspace(start = 5, stop = 30, num = 20)]
    #weight function used in prediction
    weights = ['uniform', 'distance']
    #Algorithm used to compute the nearest neighbors:
    algorithm = ['auto', 'ball_tree', 'kd_tree']
    #Leaf size passed to BallTree or KDTree
    leaf_size = [int(x) for x in np.linspace(start = 20, stop = 50, num = 15)]
    #Power parameter for the Minkowski metric
    p = [1,2,3,4,5]

    param_grid = {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p
    }
    return param_grid

# Dictionary containing all the grid functions
# Can call specific function based off of dict key.
'''
Example:
other_function(method):
    grid = grids(method)
    return grid
'''
def make_grid(method):
    grids = {
        "ada" : ada_paramgrid,
        'rf' : rf_paramgrid,
        'svr': svr_paramgrid,
        'gdb': gdb_paramgrid,
        'mlp': mlp_paramgrid,
        'knn': knn_paramgrid
    }
    return grids[method]()

