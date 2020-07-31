"""
Objective: Test regressors.py's functionality
"""

from new_tests.model_fixture import __unpickle_model__, file_list, delete_files
import pytest
import collections
import os

files, directory = file_list('pickle/datasplit')


@pytest.mark.parametrize('pkl', files)
@pytest.mark.parametrize('directory', [directory])
def test_sklearn_get_regressor(__unpickle_model__):
    """
    Objective: Match correct sklearn machine learning algorithm
    :param __unpickle_model__:
    :return:
    """
    model1 = __unpickle_model__
    model1.get_regressor()
    assert str(model1.regressor), "Can't call for regressor object"
    assert model1.task_type == 'regression'


@pytest.mark.parametrize('directory, pkl', [('pickle/reg', 'ada_reg.pkl')])
def test_sklearn_hypertune(__unpickle_model__):
    model1 = __unpickle_model__
    model1.reg()
    model1.run()
    assert type(model1.params) is dict
    # delete_files(model1.run_name)


