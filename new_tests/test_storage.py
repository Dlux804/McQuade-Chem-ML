"""
Objective: Test functionality of essential functions in storage.py
"""
import numpy as np
import os
import pandas as pd
import pytest
from new_tests.model_fixture import __unpickle_model__, delete_files


@pytest.mark.parametrize('directory, pkl', [('pickle/run', 'ada_run.pkl')])
def test_store(__unpickle_model__):
    """
    Objective: Check if important csv and json files are being created such as _data, _predictions and _attribute
    :param __unpickle_model__:
    :return:
    """
    model1 = __unpickle_model__
    model1.store()
    assert os.path.isfile(''.join([model1.run_name, '_data.csv'])), f"Can't find {model1.run_name + '_data.csv'}"
    assert os.path.isfile(''.join([model1.run_name, '_predictions.csv'])), \
                                                            f"Can't find {model1.run_name + '_predictions.csv'}"
    assert os.path.isfile(''.join([model1.run_name, '_attributes.json'])), \
        f"Can't find {model1.run_name + '_attributes.json'}"
    # delete_files(model1.run_name)


@pytest.mark.parametrize('directory, pkl', [('pickle/init', 'ada_lshort_init.pkl')])
def test_org_files(__unpickle_model__):
    model1 = __unpickle_model__
    model1.store()
    model1.org_files(zip_only=True)
    assert os.path.isfile(''.join([model1.run_name, '.zip'])), "Can't find a zip file"
    os.remove(''.join([model1.run_name, '.zip']))