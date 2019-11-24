from ingest import load_smiles
import unittest
import pandas as pd


class TestIngest(unittest.TestCase):
    def test_load_smiles(self):
    """
    This script is created to test if ingest.py is returning a csv and a SMILES column
    """
        csv, smiles_col = load_smiles(self, "water-energy.csv")  #Load the funtion
        assert type(csv) == pd.DataFrame  #See if csv is a dataframe
        assert type(smiles_col) == pd.Series  # See if smiles_col is a series
