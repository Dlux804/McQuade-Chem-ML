"""
Objective: Test functionality of essential functions in storage.py
"""

from core import models
from core import storage
import mock
import os


def test_store():
    filename = os.path.join(os.path.dirname(__file__), 'Lipo-short.csv')
    model1 = models.MlModel(algorithm='rf', dataset=filename, target='exp', feat_meth=[0],
                            tune=False, cv=2, opt_iter=2)
    model1.store()