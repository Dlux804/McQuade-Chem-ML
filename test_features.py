import unittest
import os
import pandas as pd
from core import features

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
        # os.chdir("..")  # Code currently starts in /core/ so move up to main dir
        print('Current Working Directory:', os.getcwd())

        # Next, test on every model. We know that the feats are different if the model is either nn or knn
        df = pd.read_csv("dataFiles/water-energy.csv")
        for i in model_name:
            if i == 'nn' or i == 'knn':
                df, num_feat, feat_time = features.featurize(df, i, num_feat=list(range(0, 6)))
                # self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                # self.assertNotEqual(selected_feat, 'rdkit2d', 'Something is wrong, i can feel it')
            else:
                df, num_feat, feat_time = features.featurize(df, i, num_feat=list(range(0, 6)))
                # self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                # self.assertListEqual(selected_feat, feat_sets, 'Something is wrong, i can feel it')


if __name__ == '__main__':
    unittest.main()
