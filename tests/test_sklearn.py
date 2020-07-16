"""
Objective: Test to see if the pipeline runs sklearn methods for tuned and un-tuned to completion
"""

from core import models

algorithm_list = ['ada', 'svr', 'rf', 'gdb']


def test_sklearn_no_tuned():
    """"""
    for algorithm in algorithm_list:
        model1 = models.MlModel(algorithm=algorithm, dataset='Lipo-short.csv', target='exp', feat_meth=[0],
                            tune=False, cv=2, opt_iter=2)
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        assert float(model1.predictions_stats['mse_avg']) > 0.5
        assert float(model1.predictions_stats['r2_avg']) < 0.8



def test_sklearn_tuned():
    """"""
    for algorithm in algorithm_list:
        model1 = models.MlModel(algorithm=algorithm, dataset='Lipo-short.csv', target='exp', feat_meth=[0],
                                tune=True, cv=2, opt_iter=2)
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        assert float(model1.predictions_stats['mse_avg']) > 0.5
        assert float(model1.predictions_stats['r2_avg']) < 0.8


