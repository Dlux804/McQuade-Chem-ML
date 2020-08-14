"""
Objective: Test train
"""

from new_tests.model_fixture import __unpickle_model__, delete_files, __assert_results__
import pytest
import pandas


# @pytest.mark.parametrize('directory, pkl', [('pickle/tuned', 'ada_hypertuned.pkl')])
# def test_train_reg(__unpickle_model__):
#     """
#     Objective: Test train_reg() function from train.py. Check if we're getting a predictions dataframe and a dictionary
#                 of average predictions results.
#     :param __unpickle_model__:
#     :return:
#     """
#     model1 = __unpickle_model__
#     model1.train_reg()
#     assert type(model1.predictions) is pandas.core.frame.DataFrame, \
#         "predictions class instance is supposed to be a dataframe"
#     assert type(model1.predictions_stats) is dict, "predictions_stats class instance is supposed to be a dictionary"
#     __assert_results__(model1.predictions_stats)
#     # delete_files(model1.run_name)