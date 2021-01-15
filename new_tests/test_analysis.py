"""
Objective: Test functions in analysis.py
"""
from core.storage import misc
import pytest
import os, glob
from new_tests.model_fixture import __unpickle_model__, delete_files, __assert_results__


# @pytest.mark.parametrize('directory, pkl', [('pickle/run', 'ada_run.pkl')])
# def test_graph(__unpickle_model__):
#     """
#     Objective: Test analysis.py's main graphing function. Check if it can produces a PVA graph and and importance-graph
#                 for one of the algorithm that has that feature
#     Note: While I could check for importance graph for other algorithms, I'm worried about file size. These files
#             can reach 1MB even with a dataset of 10 data points.
#     :param __unpickle_model__:
#     :return:
#     """
#     model1 = __unpickle_model__
#     model1.analyze()
#     assert os.path.isfile(''.join([model1.run_name, '_PVA.png'])), "No PVA graph found"
#     assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png'])), "No importance graph found"
    # delete_files(model1.run_name)


# @pytest.mark.parametrize('algorithm, data, exp, tuned, delete, directory', [('rf', 'Lipo-short.csv', 'exp', False,
#                                                                              False, True)])
# def test_pva_graph(__run_all__):
#     model1 = __run_all__
#     assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
#     delete_files(model1.run_name)


