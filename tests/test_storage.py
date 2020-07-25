"""
Objective: Test functionality of essential functions in storage.py
"""
from core.storage import misc
from core import models
import os, glob
import pandas as pd
from core.storage.storage import unpickle_model
import shutil
from main import ROOT_DIR

# change working directory to
os.chdir(ROOT_DIR)

with misc.cd('dataFiles/testdata'):
    model1 = models.MlModel(algorithm='ada', dataset='Lipo-short.csv', target='exp', feat_meth=[0], tune=False, cv=2,
                            opt_iter=2)


def test_store():
    """"""

    model1.store()
    assert os.path.isfile(''.join([model1.run_name, '_data.csv']))
    assert os.path.isfile(''.join([model1.run_name, '_attributes.json']))
    list(map(os.remove, glob.glob('*%s*' % model1.run_name)))


def test_pickle():
    """"""
    model1.featurize()
    model1.data_split(val=0.1)
    model1.reg()
    model1.run()
    model1.analyze()
    model1.pickle_model()
    from sklearn.metrics import mean_squared_error, r2_score
    model2 = unpickle_model(''.join([model1.run_name, '.pkl']))
    model2.run()
    # Make predictions
    predictions = model2.regressor.predict(model2.test_features)

    # Dataframe for replicate_model
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = model2.test_target
    pva['predicted'] = predictions
    assert r2_score(pva['actual'], pva['predicted']) < 0.8
    assert mean_squared_error(pva['actual'], pva['predicted']) > 0.5
    list(map(os.remove, glob.glob('*%s*' % model1.run_name)))


def test_org_files():
    model1.store()
    model1.org_files(zip_only=True)
    assert os.path.isfile(''.join([model1.run_name, '.zip']))
    os.remove(''.join([model1.run_name, '.zip']))
