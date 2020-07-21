"""
Objective: Test functionality of essential functions in storage.py
"""
from core.storage import misc
from core import models
from core import storage
import mock
import os


def test_store():
    model1 = models.MlModel(algorithm='rf', dataset='Lipo-short.csv', target='exp', feat_meth=[0],
                            tune=False, cv=2, opt_iter=2)
    model1.store()