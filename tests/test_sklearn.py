"""
Objective: Test to see if the pipeline runs sklearn methods for tuned and un-tuned to completion
"""

from core import models
import os

filename = os.path.join(os.path.dirname(__file__), 'Lipo-short.csv')
# Test for almost all models instead of knn. With a small dataset, knn throws a fit
algorithm_list = ['ada', 'svr', 'rf', 'gdb']


def test_sklearn_no_tuned():
    """"""
    for algorithm in algorithm_list:
        model1 = models.MlModel(algorithm=algorithm, dataset=filename, target='exp', feat_meth=[0],
                            tune=False, cv=2, opt_iter=2)
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        model1.analyze()
        assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
        os.remove(''.join([model1.run_name, '_PVA.png']))  # Delete PVA graphs
        assert float(model1.predictions_stats['mse_avg']) > 0.5  # Check for mse value
        assert float(model1.predictions_stats['r2_avg']) < 0.8  # Check for r2_avg value
        if algorithm in ['rf', 'gdb']:   # Check for variable importance graphs
            assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png']))
            os.remove(''.join([model1.run_name, '_importance-graph.png']))  # Remove graph


def test_sklearn_tuned():
    """"""
    for algorithm in algorithm_list:
        model1 = models.MlModel(algorithm=algorithm, dataset=filename, target='exp', feat_meth=[0],
                                tune=True, cv=2, opt_iter=2)
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        model1.analyze()
        assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))
        assert os.path.isfile(''.join([model1.run_name, '_checkpoint.pkl']))
        os.remove(''.join([model1.run_name, '_PVA.png']))
        os.remove(''.join([model1.run_name, '_checkpoint.pkl']))
        assert float(model1.predictions_stats['mse_avg']) > 0.5
        assert float(model1.predictions_stats['r2_avg']) < 0.8
        if algorithm in ['rf', 'gdb']:
            assert os.path.isfile(''.join([model1.run_name, '_importance-graph.png']))
            os.remove(''.join([model1.run_name, '_importance-graph.png']))



