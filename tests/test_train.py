"""
Objective: Test train
"""

from tests.model_fixture import __data_split_model__, delete_files, __assert_results__
import pytest
import pandas
from main import ROOT_DIR
import os
# change working directory to
os.chdir(ROOT_DIR)
algorithm_list = ['ada', 'svr', 'rf', 'gdb', 'mlp', 'nn']


@pytest.mark.parametrize('algorithm', algorithm_list)
@pytest.mark.parametrize('data, exp, tuned,directory', [('Lipo-short.csv', 'exp', True, True)])
def test_train_reg(__data_split_model__):
    model1 = __data_split_model__
    model1.reg()
    model1.make_grid()
    model1.hyperTune()
    model1.train_reg()
    assert type(model1.predictions) is pandas.core.frame.DataFrame
    assert type(model1.predictions_stats) is dict
    __assert_results__(model1.predictions_stats)
    delete_files(model1.run_name)