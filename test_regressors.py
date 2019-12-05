import pandas as pd
import features
import mock
import regressors
from models import MlModel
from sklearn.ensemble import RandomForestRegressor
import grid
import pytest

# Set up data
@pytest.fixture
def setup():
    """
    A few functions in analysis.py require setting up the same data (dataframe, target column, feature columns)
    Using pytest.fixture, we only need to set up data once for every test that need it.
    """
    # Load in data
    model_test = MlModel('rf', 'water-energy.csv', 'expt')
    # Get feature. I use rdkit2d as it is fast to generate
    df, num_feat, feat_time = features.featurize(model_test.data, model_test.algorithm, [0])
    # Split the data
    train_features, test_features, train_target, test_target, feature_list = features.targets_features(df, 'expt')
    return train_features, test_features, train_target, test_target

# Mock the call that we want to test, which is RandomizedSearchCV
@mock.patch('regressors.RandomizedSearchCV')
def test_regressors_hypertune_randomsearch(mock_rdsearchcv, setup):
    """
    In hypertune, RandomizedSearchCV plays a major part since is the call that does the actual tuning.
    This function was designed to test whether RandomizedSearchCV is called successfully if the function hypertune is
    being used.
    """
    # Load in data
    train_features, test_features, train_target, test_target = setup
    # Call the function that we would like to test
    tuned, tune_time = regressors.hyperTune(RandomForestRegressor(), train_features, train_target,
                                            grid=grid.rf_paramgrid(), folds=2, iters=1, jobs=1)
    # See if RandomizedSearchCV is called and finished successfully once
    mock_rdsearchcv.assert_called_once()


def test_regressors_hypertune(setup):
    """
    Once we know RandomizedSearchCV is called successfully, want to make sure that the output is correct.
    """
    # Load the data
    train_features, test_features, train_target, test_target = setup
    # Call the function that we would like to test
    tuned, tune_time = regressors.hyperTune(RandomForestRegressor(), train_features, train_target,
                                            grid=grid.rf_paramgrid(), folds=2, iters=1, jobs=1)
    # Assert if tuned is a dictionary
    assert type(tuned) == dict
    # Assert if tune_time is a float
    assert type(tune_time) == float