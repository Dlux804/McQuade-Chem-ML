from ingest import load_smiles
import unittest
import pandas as pd


class TestIngest(unittest.TestCase):
    def test_load_smiles(self):
        csv, smiles_col = load_smiles(self, "water-energy.csv")
        assert type(csv) == pd.DataFrame
        assert type(smiles_col) == pd.Series



