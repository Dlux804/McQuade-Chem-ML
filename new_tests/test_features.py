"""
Objective: Test functions in features.py

"""
import pytest
from numpy.random import randint
from new_tests.model_fixture import __unpickle_model__
# SMILES from clintox.csv that has been confirmed to cause issue with rdkit
# smiles = "C/C=C\1/C(=O)N[C@H](C(=O)O[C@H]\2CC(=O)N[C@@H](C(=O)N[C@H](CSSCC/C=C2)C(=O)N1)C(C)C)C(C)C"
#
#
# @pytest.mark.parametrize('directory, pkl', [('pickle/init', 'ada_lshort_init.pkl')])
# def test_featurize(__unpickle_model__):
#     """
#     Objective: Test features.py's featurize function.
#     Intent: See if it can remove troublesome SMILES, unwanted columns and append data corectly
#     :param __unpickle_model__:
#     """
#     model1 = __unpickle_model__
#     df = model1.data.append({'smiles': smiles, 'exp': randint(0.1, 1)}, ignore_index=True)
#     model1.featurize()
#     assert len(df['smiles']) != len(model1.data['smiles']), "Faulty SMILES were not removed from data"
#     assert "RDKit2D_calculated" not in model1.data.columns, "Excess columns were not removed"
#     assert len(model1.data.columns) > len(df.columns), "Current Dataframe does not contain features"
#
#
# @pytest.mark.parametrize('directory, pkl', [('pickle/featurize', 'ada_featurize.pkl')])
# def test_val_data_split(__unpickle_model__):
#     model1 = __unpickle_model__
#     model1.data_split(val=0.1)
#     assert model1.val_percent == model1.n_val / model1.n_tot
#     assert float('%g' % model1.train_percent) == model1.n_train / model1.n_tot  # Fix Trailing 1 problem in float
#     assert "in_set" in model1.data.columns, """ No "in_set" column"""
#     assert list(model1.data.loc[model1.data['in_set'] == "val"]['smiles']), """NO "val" value in "in_set" column"""
#
#
# @pytest.mark.parametrize('directory, pkl', [('pickle/featurize', 'ada_featurize.pkl')])
# def test_data_split(__unpickle_model__):
#     model1 = __unpickle_model__
#     model1.data_split()
#     assert model1.val_percent == 0
#     assert float('%g' % model1.train_percent) == model1.n_train / model1.n_tot  # Fix Trailing 1 problem in float
#     assert "in_set" in model1.data.columns, """ No "in_set" column"""
#     assert not list(model1.data.loc[model1.data['in_set'] == "val"]['smiles']), """NO "val" value in "in_set" column"""
