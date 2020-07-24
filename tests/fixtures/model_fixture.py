"""
ObjectiveL: Fixtures for sklearn model. Fixtures are functions, which will run before each test function to which it is
                                                                                                                applied.
"""
import os, glob
from core import models
from core.storage import misc
import pytest


@pytest.fixture(scope="function")
def __sklearn_untuned__(algorithm, data, exp):
    with misc.cd('../dataFiles'):
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=False, feat_meth=[0], cv=2,
                              opt_iter=2)
        return model1


@pytest.fixture(scope="function")
def __sklearn_tuned__(algorithm, data, exp):
    with misc.cd('../dataFiles'):
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=True, feat_meth=[0], cv=2,
                                opt_iter=2)
        return model1


def __assert_results__(r2, mse, rmse):
    assert float(r2) < 0.8  # Check for r2_avg value
    assert float(mse) > 0.5  # Check for mse value
    assert float(rmse) > 0.5  # Check for mse value


@pytest.fixture(scope="function")
def __run_all__(algorithm, data, exp, tuned):
    with misc.cd('../dataFiles'):
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                              opt_iter=2)
        model1.featurize()
        model1.data_split(val=0.1)
        model1.reg()
        model1.run()
        model1.analyze()
    list(map(os.remove, glob.glob('*%s*' % model1.run_name)))
    return model1