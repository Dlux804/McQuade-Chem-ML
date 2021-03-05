"""
Objective: Test functions in analysis.py
"""
from core.storage import misc
import pytest
import os, glob
from tests.model_fixture import __run_all__, delete_files

tree_algorithm = ['rf', 'gdb']


@pytest.mark.parametrize('algorithm', tree_algorithm)
@pytest.mark.parametrize('data, exp, tuned, directory', [('Lipo-short.csv', 'exp', False, True)])
def test_var_importance(__run_all__):
    model1 = __run_all__

    assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png']))
    delete_files(model1.run_name)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('rf', 'Lipo-short.csv', 'exp', False, True)])
def test_pva_graph(__run_all__):
    model1 = __run_all__
    assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
    delete_files(model1.run_name)


