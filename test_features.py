import unittest
import pandas as pd
import features
from unittest import mock



# Expected features
feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
             'morganchiral3counts', 'atompaircounts']
model_name = ['rf', 'svr', 'gdb', 'ada', 'nn', 'knn']


class TestFetures(unittest.TestCase):

    def test_featureize_nordkit2d(self):
        """
        This script was designed to test the script features.py and its function "featurize"
        The next version will use mock.patch to speed up the testing process

        """

        # Next, test on every model. We know that the feats are different if the model is either nn or knn
        df = pd.read_csv("water-energy.csv")
        for i in model_name:
            if i == 'nn' or i == 'knn':
                df, selected_feat = features.featurize(df, i, selected_feat=[0])
                self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                self.assertNotEqual(selected_feat, 'rdkit2d', 'Something is wrong, i can feel it')
            else:
                df, selected_feat = features.featurize(df, i, selected_feat=list(range(0, 7)))
                self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                self.assertListEqual(selected_feat, feat_sets, 'Something is wrong, i can feel it')


if __name__ == '__main__':
    unittest.main()
