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
"""
Pytest's fixture provides a fixed baseline so that tests execute reliably and produce consistent, repeatable, results.
They can be accessed by tests functions and can be given different parameters depending on the test
"""


@pytest.fixture(scope="function")
def __model_object__(algorithm, data, exp, tuned, directory):
    """
    Objective: return MLModel class object with custom test inputs for experimentation
    Intention: I want to have access to a model object to test for earlier functions in the `core` pipeline such as
                featurize, data_split
    :param algorithm:
    :param data:
    :param exp:
    :param tuned:
    :param directory: Whether to get data from the test directory or not. This was created in case I use data from
                        somewhere else.
    :return: model class object with only initialized model instances (instances in __init__)
    """
    if directory:
        with misc.cd('dataFiles/testdata'):
            model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                    opt_iter=2)
    else:
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                opt_iter=2)

    return model1


@pytest.fixture(scope="function")
def __data_split_model__(algorithm, data, exp, tuned, directory):
    """
    Objective: Return MLModel class object after the featurization and data split process
    Intention: Most of the process starting from choosing algorithm in the `core` pipeline requires the data to be split
                This fixture was built so that I don't have to initialize featurization and data splitting in the
                test functions
    :param algorithm:
    :param data:
    :param exp:
    :param tuned:
    :param directory: Whether to get data from the test directory or not. This was created in case I use data from
                        somewhere else.
    :return: MLModel class object after the featurization and data split process
    """
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
    """
    Objective: Run all
    :param algorithm:
    :param data:
    :param exp:
    :param tuned:
    :param delete:
    :param directory:
    :return:
    """
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