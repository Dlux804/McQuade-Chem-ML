"""
Objective: Test grid.py functionality

"""

from tests.model_fixture import  __data_split_model__
import pytest
algorithm_list = ['ada', 'svm', 'rf', 'gdb', 'mlp', 'knn', 'nn']


@pytest.mark.parametrize('algorithm', algorithm_list)
@pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', True, True)])
def test_make_grid(__data_split_model__):
    model1 = __data_split_model__
    model1.make_grid()
    assert type(model1.param_grid) is dict
