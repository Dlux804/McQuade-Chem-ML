"""
Objective: Test to see if the pipeline runs sklearn methods for tuned and un-tuned to completion
"""
import pytest
import os, glob
from tests.model_fixture import __run_all__, delete_files
# Test for almost all models instead of knn. With a small dataset, knn throws a fit

# algorithm_list = ['ada', 'svm', 'rf', 'gdb']
#
#
# @pytest.mark.parametrize('algorithm', algorithm_list)
# @pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', False, True)])
# def test_sklearn_untuned(__run_all__):
#     """"""
#     # for algor in algorithm_list:
#     model1 = __run_all__
#
#
# @pytest.mark.parametrize('algorithm', algorithm_list)
# @pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', True, True)])
# def test_sklearn_tuned(__run_all__):
#     """"""
#     # for algor in algorithm_list:
#     model1 = __run_all__
    # r2 = model1.predictions_stats['r2_avg']
    # mse = model1.predictions_stats['mse_avg']
    # rmse = model1.predictions_stats['rmse_avg']
    # __assert_results__(r2, mse, rmse)
    # assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
    # assert os.path.isfile(''.join([model1.run_name, '_checkpoint.pkl']))
    # # assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png']))





