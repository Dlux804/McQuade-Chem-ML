"""
Objective: The goal of this script is to create nodes in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from core import fragments

# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")

# TODO Add Molecular Fragments


def prep(self):
    """
    Calculate Node's properties that can't be obtained directly from the pipeline
    """
    smiles_list = list(self.data['smiles'])  # List of all SMILES
    canonical_smiles = fragments.canonical_smiles(smiles_list)  # SMILES to Canonical
    val_size = len(self.target_array) * self.val_percent  # Amount of molecules in validate dataset
    train_size = len(self.target_array) * self.train_percent  # Amount of molecules in train dataset
    test_size = len(self.target_array) * self.test_percent  # Amount of molecules in test dataset
    pva = self.predictions
    predicted = list(pva['pred_avg'])  # List of predicted value for test molecules
    test_mol = list(pva['smiles'])  # List of test molecules
    r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
    feature_length = len(self.feature_list)  # Total amount of features
    return train_size, test_size, pva, r2, mse, rmse, feature_length, val_size, canonical_smiles, predicted, test_mol


def nodes(self):
    """
    Create Neo4j nodes. Merge them if they already exist
    """
    print("Creating Nodes for %s" % self.run_name)
    train_size, test_size, pva, r2, mse, rmse, feature_length, val_size, canonical_smiles, \
                                                                                        predicted, test_mol = prep(self)
    # Make algorithm node
    if self.algorithm == "nn":
        algor = Node("Algorithm", name=self.algorithm, source="Keras", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")
    else:
        algor = Node("Algorithm", name=self.algorithm, source="sklearn", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")

    # Make FeatureMethod node
    for feat in self.feat_method_name:
        feature_method = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_method, "FeatureMethod", "feature")

    # Make Tuner node
    if self.tuned:
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
                                tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best,
                                steps=self.opt_iter)
        g.merge(tuning_algorithm, "TuningAlg", "algorithm")
    else:
        tuning_algorithm = Node("NotTuned", name="NotTuned")
        g.merge(tuning_algorithm, "NotTuned", "name")

    # Make MLModel nodes
    model = Node("MLModel", name=self.run_name, feat_time=self.feat_time, date=self.date, train_time=self.tune_time,
                 test_time=self.predictions_stats["time_avg"])
    g.create(model)

    # Make FeatureList node
    feature_list = Node("FeatureList", name="FeatureList", num=len(self.feature_list))
    g.merge(feature_list, "FeatureList", "num")

    # Make TrainSet node
    training_set = Node("TrainSet", trainsize=train_size, name="TrainSet")
    g.merge(training_set, "TrainSet",  "trainsize")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")

    # Since we can't use merge with multiple properties, I will merge RandomSplit nodes later on
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed,
                       val_percent=self.val_percent)
    g.create(randomsplit)

    # Make TestSet node
    testset = Node("TestSet", name="TestSet", RMSE=rmse, mse=mse, r2=r2, testsize=test_size)
    g.merge(testset, "TestSet", "RMSE")

    # Make ValidateSet node
    if self.val_percent > 0:
        valset = Node("ValidateSet", name="ValidateSet", valsize=val_size)
        g.merge(valset, "ValidateSet", "valsize")
    else:
        pass

    # Make nodes for all the SMILES
    for smiles, target in zip(canonical_smiles, list(self.target_array)):
        record = g.run("""MATCH (n:SMILES {SMILES:"%s"}) RETURN n""" % smiles)
        if len(list(record)) > 0:
            print(f"This SMILES, {smiles}, already exists. Updating its properties and relationships")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = mol.measurement + $target, "
                       "mol.dataset = mol.dataset + $dataset",
                       parameters={'smiles': smiles, 'target': target, 'dataset': self.dataset})
        else:
            mol = Node("SMILES", SMILES=smiles)
            g.merge(mol, "SMILES", "SMILES")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = [$target],"
                       "mol.dataset = [$dataset]",
                       parameters={'smiles': smiles, 'target': target, 'dataset': self.dataset})
