"""
Objective: Test to see if the pipeline runs keras methods for tuned and un-tuned to completion
"""
from core import models
import os
from core.storage import misc


def test_keras_no_tuned():
    """"""
    model1 = models.MlModel(algorithm='nn', dataset='Lipo-short.csv', target='exp', feat_meth=[0], tune=False, cv=2,
                                opt_iter=2)
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
    assert os.path.isfile(''.join([model1.run_name, '.h5']))  # Check for PVA graphs
    os.remove(''.join([model1.run_name, '.h5']))
    os.remove(''.join([model1.run_name, '_PVA.png']))  # Delete PVA graphs
    assert float(model1.predictions_stats['mse_avg']) > 0.5  # Check for mse value
    assert float(model1.predictions_stats['r2_avg']) < 0.8  # Check for r2_avg value


def test_keras_tuned():
    """"""
    model1 = models.MlModel(algorithm='nn', dataset='Lipo-short.csv', target='exp', feat_meth=[0], tune=True, cv=2,
                            opt_iter=2)
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    assert os.path.isfile(''.join([model1.run_name, '_PVA.png']))  # Check for PVA graphs
    assert os.path.isfile(''.join([model1.run_name, '.h5']))  # Check for PVA graphs
    assert os.path.isfile(''.join([model1.run_name, '_checkpoint.pkl']))
    os.remove(''.join([model1.run_name, '_checkpoint.pkl']))
    os.remove(''.join([model1.run_name, '.h5']))
    os.remove(''.join([model1.run_name, '_PVA.png']))  # Delete PVA graphs
    assert float(model1.predictions_stats['mse_avg']) > 0.5  # Check for mse value
    assert float(model1.predictions_stats['r2_avg']) < 0.8  # Check for r2_avg value
