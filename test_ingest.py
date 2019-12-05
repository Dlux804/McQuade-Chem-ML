from ingest import load_smiles
# import unittest
import pandas as pd
from models import MlModel
import mock

# Test load_smiles when drop = False
@mock.patch('models.MlModel')  # Using mock to "replace" more time demanding calls like MlModel
def test_load_smiles_dropfalse(mock_mlmodel):
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=False
    Using mocker (a pytest plug-in), we're able to to test for the necessary variable without having to make time intensive calls
    https://tech.comtravo.com/testing/Testing_Machine_Learning_Models_with_Unittest/
    I'm still testing on how to use mock more efficiently
    """
    model_test = MlModel('rf', 'water-energy.csv', 'expt')  # Calling MlModel to get our class instances
    # csv, smiles_col = load_smiles(model_test.data, "water-energy.csv", drop=False)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column
    mock.patch.stopall()  # Stop mock object

@mock.patch('models.MlModel')  # Using mock to "replace" more time demanding calls like MlModel
def test_load_smiles_droptrue(mock_mlmodel):
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=True
    Using mocker (a pytest plug-in), we're able to to test for the necessary variable without having to make time intensive calls
    https://tech.comtravo.com/testing/Testing_Machine_Learning_Models_with_Unittest/
    I'm still testing on how to use mock more efficiently
    """
    model_test = MlModel('gdb', 'water-energy.csv', 'expt', drop=False)  # Calling MlModel to get our class instances, drop=False in this case
    # csv, smiles_col = load_smiles(self, "water-energy.csv", drop=True)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column
    mock.patch.stopall()  # Stop mock object

