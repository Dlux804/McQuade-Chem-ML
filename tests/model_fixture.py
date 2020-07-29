"""
ObjectiveL: Fixtures for sklearn model. Fixtures are functions, which will run before each test function to which it is
                                                                                                                applied.
"""
import os
import glob
from core import models
from core.storage import misc
import pytest
from main import ROOT_DIR

# change working directory to
os.chdir(ROOT_DIR)


@pytest.fixture(scope="function")
def __model_object__(algorithm, data, exp, tuned, delete, directory):

    if directory:
        with misc.cd('dataFiles/testdata'):
            model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                    opt_iter=2)
    else:
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                opt_iter=2)
    if delete:
        delete_files(model1.run_name)
    else:
        pass
    return model1


@pytest.fixture(scope="function")
def __data_split_model__(algorithm, data, exp, tuned, directory):

    if directory:
        with misc.cd('dataFiles/testdata'):
            model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                    opt_iter=2)
    else:
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                opt_iter=2)
    model1.featurize()
    model1.data_split(val=0.1)

    return model1


@pytest.fixture(scope="function")
def __run_all__(algorithm, data, exp, tuned, delete, directory):
    if directory:
        with misc.cd('dataFiles/testdata'):
            model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                    opt_iter=2)
    else:
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                opt_iter=2)
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    if delete:
        delete_files(model1.run_name)
    else:
        pass
    return model1


def __assert_results__(predictions_stats):
    assert float(predictions_stats['r2_avg'])
    assert float(predictions_stats['mse_avg'])
    assert float(predictions_stats['rmse_avg'])


def delete_files(run_name):
    return list(map(os.remove, glob.glob('*%s*' % run_name)))


### This secion is created for acquiring datatypes before testing
# with misc.cd('dataFiles/testdata'):
#     model1 = models.MlModel(algorithm='gdb', dataset='ESOL-short.csv', target='exp', tune=True, feat_meth=[0], cv=2,
#                             opt_iter=2)
#     model1.featurize()
#     model1.data_split(val=0.1)
#     model1.reg()
#     model1.make_grid()
#     model1.hyperTune()
#     print(type(model1.params))
#     delete_files(model1.run_name)