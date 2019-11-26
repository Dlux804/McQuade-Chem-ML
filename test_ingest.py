from ingest import load_smiles
# import unittest
import pandas as pd
from models import MlModel
import models


def test_load_smiles_dropfalse(mocker):
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=False
    Using mocker (a pytest plug-in), we're able to to test for the necessary variable without having to make time intensive calls
    https://tech.comtravo.com/testing/Testing_Machine_Learning_Models_with_Unittest/
    """
    mocker.patch('models.MlModel')  # Using mocker to replace the class method with a no-op object that can be called but does not do anything and typically has no side effects
    model_test = MlModel('rf', 'water-energy.csv', 'expt')  # Calling MlModel to get our class instances
    # csv, smiles_col = load_smiles(model_test.data, "water-energy.csv", drop=False)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column
    mocker.resetall()  #Reset all mocked object


def test_load_smiles_droptrue(mocker):
    """
    Test if ingest.py is returning a dataframe and a SMILES column when drop=True
    Using mocker (a pytest plug-in), we're able to to test for the necessary variable without having to make time intensive calls
    https://tech.comtravo.com/testing/Testing_Machine_Learning_Models_with_Unittest/

    """
    mocker.patch('models.MlModel')  # Using mocker to replace the class method with a no-op object that can be called but does not do anything and typically has no side effects
    model_test = MlModel('gdb', 'water-energy.csv', 'expt', drop=False)  # Calling MlModel to get our class instances, drop=False in this case
    # csv, smiles_col = load_smiles(self, "water-energy.csv", drop=True)
    assert type(model_test.data) == pd.DataFrame  # Testing to see if data is a dataframe
    assert type(model_test.smiles) == pd.Series  # Testing to see if we get a SMILES column
    mocker.resetall()  #Reset all mocked object


