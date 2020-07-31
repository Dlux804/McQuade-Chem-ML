"""
Objective: Test ingest.py's functionality
"""

from tests.model_fixture import  __model_object__
import pytest
import pandas as pd


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('rf', 'Lipo-short.csv', 'exp', False, True)])
def test_ingest(__model_object__):
    model1 = __model_object__
    assert type(model1.data) is pd.core.frame.DataFrame
    assert type(model1.smiles_series) is pd.Series
    assert len(model1.data.columns) == 2
