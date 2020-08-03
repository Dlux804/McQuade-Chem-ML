"""
Objective: Test to see if the pipeline runs keras methods for tuned and un-tuned to completion
"""

import pytest
from new_tests.model_fixture import __run_all__, delete_files


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', False, True)])
def test_nn_untuned(__run_all__):
    """
    Objective: Run an untuned keras model from start to get prediction results.
   Note: Since I can't unload a h5 file to get important information for almost all tests, I decided to test its
            functionality by running a keras model from start to getting prediction results
    :param __run_all__:
    :return:
    """
    # for algor in algorithm_list:
    model1 = __run_all__
    delete_files(model1.run_name)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', True, True)])
def test_nn_tuned(__run_all__):
    """
    Objective: Run a tuned keras model from start to get prediction results.
    Note: Since I can't unload a h5 file to get important information for almost all tests, I decided to test its
            functionality by running a keras model from start to getting prediction results
    :param __run_all__:
    :return:
    """
    # for algor in algorithm_list:
    model1 = __run_all__
    delete_files(model1.run_name)