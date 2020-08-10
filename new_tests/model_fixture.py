"""
ObjectiveL: Fixtures for sklearn model. Fixtures are functions, which will run before each test function to which it is
                                                                                                                applied.
"""
import os
import glob
from core import models
from core.storage import misc
import pytest
from core.storage.misc import unpickle_model
# from main import ROOT_DIR
# # change working directory to
# os.chdir(ROOT_DIR)
"""
Pytest's fixture provides a fixed baseline so that tests execute reliably and produce consistent, repeatable, results.
They can be accessed by tests functions and can be given different parameters depending on the test
"""


@pytest.fixture(scope="function")
def __run_all__(algorithm, data, exp, tuned, directory):
    """
    Objective: Run all
    :param algorithm:
    :param data:
    :param exp:
    :param tuned:
    :param directory:
    :return:
    """
    if directory:
        with misc.cd('../dataFiles/testdata'):
            model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                    opt_iter=2)
    else:
        model1 = models.MlModel(algorithm=algorithm, dataset=data, target=exp, tune=tuned, feat_meth=[0], cv=2,
                                opt_iter=2)
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    return model1


def file_list(directory):
    """
    Objective: Return list of files in a directory
    Intent: Use for test that requires list of files
    :param directory: desired directory. The directory here has to match with the directory in __unpickle_model__ when
                                        being used in the same function
    :return: list: list of files
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))], directory


@pytest.fixture(scope="function")
def __unpickle_model__(directory, pkl):
    """
    Objective: Unpickle pickle file to retrieve model object for further testing
    :param pkl: str: pkl file
    :return: class object: MLModel class object
    """
    with misc.cd(directory):
        model1 = unpickle_model(''.join([pkl]))
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