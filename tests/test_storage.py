"""
Objective: Test functionality of essential functions in misc.py
"""
import numpy as np
import os
import pandas as pd
from core.storage.misc import unpickle_model
import pytest
from main import ROOT_DIR
from tests.model_fixture import __run_all__, __model_object__, delete_files, __assert_results__, __run_all__without_analyze
# change working directory to
os.chdir(ROOT_DIR)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('rf', 'Lipo-short.csv', 'exp', False, True)])
def test_store(__model_object__):
    """"""
    model1 = __model_object__
    model1.store()
    assert os.path.isfile(''.join([model1.run_name, '_data.csv']))
    assert os.path.isfile(''.join([model1.run_name, '_attributes.json']))
    delete_files(model1.run_name)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('rf', 'Lipo-short.csv', 'exp', True, True)])
def test_pickle(__run_all__without_analyze):
    """"""
    model1 = __run_all__without_analyze
    model1.pickle_model()
    from sklearn.metrics import mean_squared_error, r2_score
    model2 = unpickle_model(''.join([model1.run_name, '.pkl']))
    model2.run()
    # Make predictions
    predictions = model2.estimator.predict(model2.test_features)

    # Dataframe for replicate_model
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = model2.test_target
    pva['predicted'] = predictions
    assert r2_score(pva['actual'], pva['predicted']) < 0.9
    assert mean_squared_error(pva['actual'], pva['predicted']) > 0.2
    assert np.sqrt(mean_squared_error(pva['actual'], pva['predicted'])) > 0.2
    delete_files(model1.run_name)


@pytest.mark.parametrize('algorithm, data, exp, tuned, directory', [('rf', 'Lipo-short.csv', 'exp', False, True)])
def test_org_files(__model_object__):
    model1 = __model_object__
    model1.store()
    model1.org_files(zip_only=True)
    assert os.path.isfile(''.join([model1.run_name, '.zip']))
    os.remove(''.join([model1.run_name, '.zip']))