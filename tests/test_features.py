import pandas as pd
import numpy as np
import mock
import os, sys


# before importing local modules, must add root dir to system path
# capture location of current file (/root/tests/)
myPath = os.path.dirname(os.path.abspath(__file__))
# add to system path the root dir with relative notation: /../ (go up one dir)
sys.path.insert(0, myPath + '/../')

from core import features, misc
from main import ROOT_DIR

# Expected features
feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
             'morganchiral3counts', 'atompaircounts']
# Expected models
model_name = ['rf', 'svr', 'gdb', 'ada', 'nn', 'knn']


# Mock the function MakeGenerator since it takes a lot of time to generate and it's the thing we want to test
@mock.patch('core.features.MakeGenerator')
def test_features_makegenerator(mock_makegenerator):
    """
    Since MakeGenerator is the main part of this function, we should test to see if it is called successfully. This will
    help us narrow down the actual error if we ever run into one.
    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        # Call function featurize
        df, num_feat, feat_time = features.featurize(df, 'rf', feat_meth=[0])  # Call function featurize
        # Test to see if MakeGenerator is called
        mock_makegenerator.assert_called()
        # Stop mock
        mock.patch.stopall()


def test_featurize_remove():
    """
    We want to test if the first descriptor generated different between using ML models that need normalization and ones
    that don't
    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        # Call function featurize using nn
        df1, num_feat1, feat_time1 = features.featurize(df, 'nn', feat_meth=[0])
        # Call function featurize using rf
        df2, num_feat2, feat_time2 = features.featurize(df, 'rf', feat_meth=[0])
        # See if the two dataframes are the same
        assert df1.equals(df2) == True

def test_featurize():
    """
    This function was designed to test the script features.py and its function "featurize"

    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        # Next, test on every model. We know that the feats are different if the model is either nn or knn
        for i in model_name:
            # Exception with nn and knn
            if i == 'nn' or i == 'knn':
                df, selected_feat, feat_time = features.featurize(df, i, feat_meth=[0])  # Call function featurize
                # Test to see if df is a dataframe
                assert type(df) == pd.DataFrame, 'Something is wrong, i can feel it'
                # Test to see if the variable time is a number
                assert type(feat_time) == float, 'Something is wrong, i can feel it'
            else:
                df, selected_feat, feat_time = features.featurize(df, i, feat_meth=[0])
                # Test to see if df is a dataframe
                assert type(df) == pd.DataFrame, 'Something is wrong, i can feel it'
                # Test to see if the variable time is a number
                assert type(feat_time) == float, 'Something is wrong, i can feel it'

def test_targets_features():
    """
    This function was designed to test the script features.py and its function "targets_features"

    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        print('Now in:', os.getcwd())
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        target = df['expt']  # Target column
        feature = df.drop(['expt', 'smiles'], axis=1)  # All feature columns
        # Split the data
        train_features, test_features, train_target, test_target, feature_list = features.data_split(df, 'expt')
        # Test to see if we get a 20% split on test features
        assert np.round(test_features.shape[0] / feature.shape[0] * 100, -1) == 20.0
        # Test to see if we get a 20% split on test target
        assert np.round(test_target.shape[0] / target.shape[0] * 100, -1) == 20.0
        # Test to see if feature_list is a list
        assert type(feature_list) == list