"""
Objective: Test to see if the pipeline runs keras methods for tuned and un-tuned to completion
"""

import pytest
from tests.model_fixture import __run_all__without_analyze


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', False, True)])
def test_sklearn_untuned(__run_all__without_analyze):
    """"""
    # for algor in algorithm_list:
    model1 = __run_all__without_analyze


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('nn', 'Lipo-short.csv', 'exp', True, True)])
def test_sklearn_tuned(__run_all__without_analyze):
    """"""
    # for algor in algorithm_list:
    model1 = __run_all__without_analyze

