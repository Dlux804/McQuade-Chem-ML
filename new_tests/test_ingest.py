"""
Objective: Test ingest.py's functionality
"""
from new_tests.model_fixture import __unpickle_model__, file_list
import pytest
import pandas as pd
from core.models import rds, cds  # regression and classification data list

files = file_list('pickle/init')


@pytest.mark.parametrize('pkl', files)
@pytest.mark.parametrize('directory', ['pickle/init'])
def test_load_smiles(__unpickle_model__):
    """
    Objective: Test ingest.py's load_smiles

    :param __unpickle_model__:
            directory: pkl file with MLModel class object add init stage
            pkl: pickle file
    """
    model1 = __unpickle_model__
    assert type(model1.data) is pd.core.frame.DataFrame  # Check if model1.data is a dataframe
    assert type(model1.smiles_series) is pd.Series  # Check if smiles_series instance is pandas Series object
    if model1.dataset in rds or model1.dataset in cds:
        assert len(model1.data.columns) == 2  # Check number of columns in data instance if it's a regression data
    else:
        assert len(model1.data.columns) > 2
