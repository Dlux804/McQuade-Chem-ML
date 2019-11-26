import unittest
import pandas as pd
import features
import numpy as np


# Expected features
feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
             'morganchiral3counts', 'atompaircounts']
model_name = ['rf', 'svr', 'gdb', 'ada', 'nn', 'knn']


class TestFeatures():

    def test_featurize(self):
        """
        This function was designed to test the script features.py and its function "featurize"

        """
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        # Next, test on every model. We know that the feats are different if the model is either nn or knn
        for i in model_name:
            # Exception with nn and knn
            if i == 'nn' or i == 'knn':
                df, selected_feat, feat_time = features.featurize(df, i, num_feat=[0])  # Call function featurize
                assert type(df) == pd.DataFrame, 'Something is wrong, i can feel it'  # Test to see if df is a dataframe
                assert 'rdkit2d' not in selected_feat, 'Something is wrong, i can feel it'  # Test to see if 'rdkit2d' is still in the feature list
                assert type(feat_time) == float, 'Something is wrong, i can feel it'  # Test to see if the variable time is a number
            else:
                df, selected_feat, feat_time = features.featurize(df, i, num_feat=[0])
                assert type(df) == pd.DataFrame, 'Something is wrong, i can feel it'  # Test to see if df is a dataframe
                assert 'rdkit2d' in selected_feat, 'Something is wrong, i can feel it'  # Test to see if 'rdkit2d' is in the feature list
                assert type(feat_time) == float, 'Something is wrong, i can feel it'  # Test to see if the variable time is a number

    def test_targets_features(self):
        """
        This function was designed to test the script features.py and its function "targets_features"

        """
        # Read csv into dataframe
        df = pd.read_csv("water-energy.csv")
        target = df['expt']  # Target column
        feature = df.drop(['expt', 'smiles'], axis=1)  # All feature columns
        train_features, test_features, train_target, test_target, feature_list = features.targets_features(df, 'expt')  # Split the data
        assert np.round(test_features.shape[0] / feature.shape[0] * 100, -1) == 20.0  # Test to see if we get a 20% split on test features
        assert np.round(test_target.shape[0] / target.shape[0] * 100, -1) == 20.0  # Test to see if we get a 20% split on test target
        assert type(feature_list) == list  # Test to see if feature_list is a list