from ingest import load_smiles
import unittest
import pandas as pd


class TestIngest(unittest.TestCase):
    def test_load_smiles_dropfalse(self):
        """
        This script was made to test if ingest.py is returning a dataframe and a SMILES column when drop=False

        """
        csv, smiles_col = load_smiles(self, "water-energy.csv", drop=False)
        self.assertEqual(type(csv), pd.DataFrame, 'Something is wrong, i can feel it')
        self.assertEqual(type(smiles_col), pd.Series, 'Something is wrong, i can feel it')
        # assert type(csv) == pd.DataFrame
        # assert type(smiles_col) == pd.Series

    def test_load_smiles_droptrue(self):
        """
        This script was made to test if ingest.py is returning a dataframe and a SMILES column when drop=True

        """
        csv, smiles_col = load_smiles(self, "water-energy.csv", drop=True)
        self.assertEqual(type(csv), pd.DataFrame, 'Something is wrong, i can feel it')
        self.assertEqual(type(smiles_col), pd.Series, 'Something is wrong, i can feel it')

