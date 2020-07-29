"""
Objective: Test regressors.py's functionality
"""

from tests.model_fixture import __data_split_model__, delete_files
import pytest
import collections
import os
from main import ROOT_DIR
# change working directory to
os.chdir(ROOT_DIR)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', False, True)])
def test_nn_get_regressor(__data_split_model__):
    model1 = __data_split_model__
    model1.get_regressor()
    assert model1.regressor, "Can't call for regressor object"
    assert type(model1.fit_params) is dict, "fit params is not a dictionary"
    assert model1.task_type == 'regression'


sklearn_list = ['ada', 'svr', 'rf', 'gdb', 'mlp', 'knn']


@pytest.mark.parametrize('algorithm', sklearn_list)
@pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', False, True)])
def test_sklearn_get_regressor(__data_split_model__):
    model1 = __data_split_model__
    model1.get_regressor()
    assert model1.task_type == 'regression'
    assert str(model1.regressor)


sklearn_list.remove('knn')


@pytest.mark.parametrize('algorithm', sklearn_list)
@pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', True, True)])
def test_sklearn_hypertune(__data_split_model__):
    model1 = __data_split_model__
    model1.reg()
    model1.make_grid()
    model1.hyperTune()
    assert type(model1.params) is dict
    delete_files(model1.run_name)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', True, True)])
def test_nn_hypertune(__data_split_model__):
    model1 = __data_split_model__
    model1.reg()
    model1.make_grid()
    model1.hyperTune()
    assert type(model1.params) is dict
    assert os.path.isfile(''.join([model1.run_name, '.h5']))  # Check for PVA graphs
    delete_files(model1.run_name)
