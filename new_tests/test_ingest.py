"""
Objective: Test ingest.py's functionality
"""
from new_tests.model_fixture import __unpickle_model__, file_list
import pytest
import pandas as pd
from core.models import rds, cds  # regression and classification data list
files, directory = file_list('pickle/init')


@pytest.mark.parametrize('pkl', files)
@pytest.mark.parametrize('directory', [directory])
def test_load_smiles(__unpickle_model__):
    """
    Objective: Test ingest.py's load_smiles. Check if daa instance is a dataframe, smiles_series instance is a
                pandas Series object and that we have the correct amount of columns depending on the dataset.

    :param __unpickle_model__:
            directory: pkl file with MLModel class object add init stage
            pkl: pickle file
    """
    model1 = __unpickle_model__
    assert type(model1.data) is pd.core.frame.DataFrame, "data class instance needs to be a pandas Dataframe"
    assert type(model1.smiles_series) is pd.Series, "smiles_series class instance needs to be a pandas Series"
    if model1.dataset in rds or model1.dataset in cds:
        assert len(model1.data.columns) == 2, "Dataframe should only have 2 columns"
    else:
        assert len(model1.data.columns) > 2, "Dataframe should have more than 2 columns"
