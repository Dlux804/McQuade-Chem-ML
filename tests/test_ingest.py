from ingest import load_smiles
# import unittest
import pandas as pd
from models import MlModel
import mock
import pandas as pd

# Mock the module Chem since it's the thing we want to test
@mock.patch('ingest.Chem')
def test_ingest_Chem(mock_chem):
    """
    Chem.MolFromSmiles is the core function used to make the function load_smiles in ingest.py. Therefore, we would like
    to test if there's any error in using this function. This will help us narrow down the actual error if we ever run
    into one.
    """
    df = pd.read_csv('water-energy.csv')
    model_test = MlModel('rf', 'water-energy.csv', 'expt')  # Calling MlModel to get our class instances
    mock_chem.MolFromSmiles.assert_called()
    mock.patch.stopall()

# Test load_smiles when drop = False
def test_load_smiles_dropfalse():
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=False
    I'm still testing on how to use mock more efficiently
    """
    model_test = MlModel('rf', 'water-energy.csv', 'expt')  # Calling MlModel to get our class instances
    # csv, smiles_col = load_smiles(model_test.data, "water-energy.csv", drop=False)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column

def test_load_smiles_droptrue():
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=True
    I'm still testing on how to use mock more efficiently
    """
    model_test = MlModel('gdb', 'water-energy.csv', 'expt', drop=False)  # Calling MlModel to get our class instances, drop=False in this case
    # csv, smiles_col = load_smiles(self, "water-energy.csv", drop=True)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column

