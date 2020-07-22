"""
Objective: Test functionality of essential functions in storage.py
"""
from core.storage import misc
from core import models
import os, glob



with misc.cd('../dataFiles'):
    model1 = models.MlModel(algorithm='ada', dataset='Lipo-short.csv', target='exp', feat_meth=[0], tune=False, cv=2,
                            opt_iter=2)

# print(glob.glob('*%s*' % model1.run_name))


def test_store():
    """"""

    model1.store()
    assert os.path.isfile(''.join([model1.run_name, '_data.csv']))
    assert os.path.isfile(''.join([model1.run_name, '_attributes.json']))

    map(os.remove, glob.glob('*%s*' % model1.run_name))


def test_pickle():
    """"""
    model1.pickle_model()



