import unittest
import os
import pandas as pd
from core import features
from core import models, misc
from main import ROOT_DIR

# Expected features
feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
             'morganchiral3counts', 'atompaircounts']
model_name = ['rf', 'svr', 'gdb', 'ada', 'nn', 'knn']


class TestFetures(unittest.TestCase):

    def test_featurize_nordkit2d(self):
        """
        This script was designed to test the script features.py and its function "featurize"
        The next version will use mock.patch to speed up the testing process
        """
        # change working directory to
        os.chdir(ROOT_DIR)
        # move to dataFiles
        with misc.cd('dataFiles'):
            print('Now in:', os.getcwd())

            # Next, test on every model. We know that the feats are different if the model is either nn or knn
            df = pd.read_csv("water-energy.csv")
            for i in model_name:
                if i == 'nn' or i == 'knn':
                    df, num_feat, feat_time = features.featurize(df, i, num_feat=list(range(0, 6)))
                    print('num_feat is:', num_feat)
                    self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                    self.assertNotEqual(num_feat, '[0]', 'Something is wrong, i can feel it')
                else:
                    df, num_feat, feat_time = features.featurize(df, i, num_feat=list(range(0, 6)))
                    self.assertEqual(type(df), pd.DataFrame, 'Something is wrong, i can feel it')
                    # self.assertListEqual(selected_feat, feat_sets, 'Something is wrong, i can feel it')


if __name__ == '__main__':
    unittest.main()
