import pandas as pd
from core import analysis, features, regressors, misc, models
import os, sys
import mock
import pytest
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from main import ROOT_DIR


# before importing local modules, must add root dir to system path
# capture location of current file (/root/tests/)
myPath = os.path.dirname(os.path.abspath(__file__))
# add to system path the root dir with relative notation: /../ (go up one dir)
sys.path.insert(0, myPath + '/../')

# Set up data
@pytest.fixture
def setup():
    """
    A few functions in analysis.py require setting up the same data (dataframe, target column, feature columns)
    Using pytest.fixture, we only need to set up data once for every test that need it.
    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Load in data
        model_test = models.MlModel('rf', 'water-energy.csv', 'expt')
        # Get feature. I use rdkit2d as it is fast to generate
        df, num_feat, feat_time = features.featurize(model_test.data, model_test.algorithm, [0])
        # Split the data
        train_features, test_features, train_target, test_target, feature_list = features.targets_features(df, 'expt')
        return train_features, test_features, train_target, test_target


# Test analysis's "predict" function
def test_analysis_predict(setup):
    """

    This function was designed to test analysis's "predict" function. We want to make sure that we're getting a dataframe
    that contains predict(float) and actual(float) results and a number for prediction time.
    """

    # Use the fixture to set up necessary data
    train_features, test_features, train_target, test_target = setup
    # Call the function we want to test
    pva, time_fit = analysis.predict(RandomForestRegressor(), train_features, test_features, train_target, test_target)
    # Assert the pva is a dataframe
    assert type(pva) == pd.DataFrame
    # Assert the predicted column in pva are floats
    assert np.array_equal(pva.predicted, pva.predicted.astype(float))
    # Assert the actual column in pva are floats
    assert np.array_equal(pva.actual, pva.actual.astype(float))
    # Assert that we have prediction time which is a float
    assert type(time_fit) == float


# Test function replitcate_model
def test_replicate_model():
    """
    This function was designed to test analysis's "replicate_model" function. We want to make sure that the function
    runs properly and a dictionary of values are given in the end
    This function uses a lot of class instance (self.something) so I have to generate data again instead of using setup.
    But since they pay me the big bucks, I can make it work.
    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Load in data
        model1 = models.MlModel('gdb', 'water-energy.csv', 'expt')
        # Generate Features
        model1.featurization([0])
        # Make a class instance called regressor
        model1.regressor = RandomForestRegressor()
        # Call the function we want to test
        stats = analysis.replicate_model(model1, 3)
        # Assert if this variable is a dictionary
        assert type(stats) == dict
        # Assert if r2_avg (should be the first key) is in the dictionary.
        assert 'r2_avg' in stats
        # Assert if 'time_std' (should be the last key) is in the dictionary.
        assert 'time_std' in stats


# Test multi_predict
def test_multi_predict(setup):
    """
    This function was designed to test analysis's "multi_predict" function. We want to make sure that the function
    runs properly and a pva dataframe(all floats) and a time(float) is returned
    """
    # Use the fixture to set up necessary data
    train_features, test_features, train_target, test_target = setup
    # Call the function we want to test
    pva, fit_time = analysis.multipredict(RandomForestRegressor(), train_features, test_features, train_target,
                                          test_target, n=5)
    # Assert the pva is a dataframe
    assert type(pva) == pd.DataFrame
    # Assert the predicted column in pva are floats
    assert np.array_equal(pva.pred_avg, pva.pred_avg.astype(float))
    # Assert the actual column in pva are floats
    assert np.array_equal(pva.pred_std, pva.pred_std.astype(float))
    # Assert that we have prediction time which is a float
    assert type(fit_time) == float


# Test pvaM_graphs
def test_pvaM_graphs(setup):
    """
    This function was designed to test analysis's "pvaM_graphs" function. We want to make sure that the function
    runs properly and a graph is returned.
    Since this function give us a graph, something that I can't really pass to a variable, I'll just test to see if this
    function is ... functioning properly (ba-dum-tish). If the variable 'fig' was passed then I would be able to use it
    to test but we don't really need to complicate things just for the sake of testing.
    """
    # Use the fixture to set up necessary data
    train_features, test_features, train_target, test_target = setup
    # Call multipredict to get the pva as input for pvaM_graphs
    pva, fit_time = analysis.multipredict(RandomForestRegressor(), train_features, test_features, train_target,
                                          test_target, n=5)
    # Call the function we want to test
    analysis.pvaM_graphs(pva)


# Test pva_graphs
def test_pva_graphs(setup):
    """
    This function was designed to test analysis's "pva_graphs" function. We want to make sure that the function
    runs properly and a graph is returned.
    Since this function give us a graph, something that I can't really pass to a variable, I'll just test to see if this
    function is ... functioning properly (ba-dum-tish). If the variable 'fig' was passed then I would be able to use it
    to test but we don't really need to complicate things just for the sake of testing.
    """
    # Use the fixture to set up necessary data
    train_features, test_features, train_target, test_target = setup
    # Generate the pva dataframe as input
    pva, time_fit = analysis.predict(RandomForestRegressor(), train_features, test_features, train_target, test_target)
    # Call the function we would like to test
    analysis.pva_graphs(pva, 'rf')