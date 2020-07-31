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
    Objective: Match correct sklearn machine learning algorithm. Check if we're getting the correct task_type
    Note: I have to turn regressor instance to string in order to call it
    :param __unpickle_model__:
    :return:
    """
    model1 = __unpickle_model__
    model1.get_regressor()
    assert str(model1.regressor), "Can't call for regressor class instance"
    assert model1.task_type == 'regression', "task_type class instance should be regression"


@pytest.mark.parametrize('directory, pkl', [('pickle/reg', 'ada_reg.pkl')])
def test_sklearn_hypertune(__unpickle_model__):
    """
    Objective: Test hyperTune function. The asserted variables are not very important. They're more like checkpoints
                on a function. These checkpoints are at important points in the function. fit_params lets us know if
                the function can distinguish between sklearn and keras algorithms. params lets us know that skopt is
                functioning correctly
    Note: In pycharm model1.params is an OrderedDict object but in CircleCI, it's a 'dict' object. This conflict is
          concerning.
    :param __unpickle_model__:
    :return:
    """
    model1 = __unpickle_model__
    model1.reg()
    model1.run()
    assert model1.fit_params is None, "fit_params class instance should be None object"
    assert type(model1.params) is dict, "params class instance should be a dictionary"
    # assert type(model1.params) is collections.OrderedDict
    # delete_files(model1.run_name)


