"""
Objective: Test grid.py functionality

"""

from new_tests.model_fixture import __unpickle_model__, file_list
import pytest

files = file_list('pickle/datasplit')


@pytest.mark.parametrize('pkl', files)
@pytest.mark.parametrize('directory', ['pickle/datasplit'])
def test_make_grid(__unpickle_model__):
    model1 = __unpickle_model__
    model1.make_grid()
    assert type(model1.param_grid) is dict
