"""
Objective: Test functions in analysis.py
"""

import pytest
import os, glob
from tests.fixtures.model_fixture import __sklearn_untuned__, __sklearn_tuned__, __assert_results__

tree_algorithm = ['rf', 'gdb']


@pytest.mark.parametrize('algorithm', tree_algorithm)
@pytest.mark.parametrize('data, exp', [('Lipo-short.csv', 'exp')])
def test_var_importance(__sklearn_tuned__):
    model1 = __sklearn_tuned__
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png']))
    list(map(os.remove, glob.glob('*%s*' % model1.run_name)))


@pytest.mark.parametrize('algorithm, data, exp', [('rf', 'Lipo-short.csv', 'exp')])
def test_var_importance(__sklearn_untuned__):
    model1 = __sklearn_untuned__
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
    list(map(os.remove, glob.glob('*%s*' % model1.run_name)))


