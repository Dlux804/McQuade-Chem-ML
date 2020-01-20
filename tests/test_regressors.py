import pandas as pd
import mock
from sklearn.ensemble import RandomForestRegressor
import pytest
import os, sys

# before importing local modules, must add root dir to system path
# capture location of current file (/root/tests/)
myPath = os.path.dirname(os.path.abspath(__file__))
# add to system path the root dir with relative notation: /../ (go up one dir)
sys.path.insert(0, myPath + '/../')

# Now we can import modules from other directory
from core import features, regressors, models, grid, misc
from main import ROOT_DIR



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

# Mock the call that we want to test, which is RandomizedSearchCV
@mock.patch('core.regressors.RandomizedSearchCV')
def test_regressors_hypertune_randomsearch(mock_rdsearchcv, setup):
    """
    In hypertune, RandomizedSearchCV plays a major part since is the call that does the actual tuning.
    This function was designed to test whether RandomizedSearchCV is called successfully if the function hypertune is
    being used.
    """
    # Load in data
    train_features, test_features, train_target, test_target = setup
    # Call the function that we would like to test
    regressors.hyperTune(RandomForestRegressor(), train_features, train_target,
                                            grid=grid.rf_paramgrid(), folds=2, iters=1, jobs=1)
    # See if RandomizedSearchCV is called
    mock_rdsearchcv.assert_called_once()
    mock.patch.stopall()



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
