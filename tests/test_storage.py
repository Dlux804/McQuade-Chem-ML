"""
Objective: Test functionality of essential functions in storage.py
"""
from core.storage import misc
from core import models
from core.storage import storage
import pytest
import os
import pandas as pd
with misc.cd('../tests'):
    print(os.getcwd())

def test_store():
    """"""
    with misc.cd('../tests'):
        model1 = models.MlModel(algorithm='rf', dataset='Lipo-short.csv', target='exp', feat_meth=[0], tune=False, cv=2,
                                opt_iter=2)



