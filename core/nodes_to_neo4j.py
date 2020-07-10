"""
Objective: The goal of this script is to create nodes in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from core import fragments
from tqdm import tqdm

# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def prep(self):
    """
    Objective: Calculate or prepare variables that we don't have class instances for
    Intent: I want to have one function that calculates the data I need for the ontology. I don't think we need
            class instances for these data since they can be easily obtained with one ot two lines of code each.
    """
    smiles_list = list(self.data['smiles'])  # List of all SMILES
    canonical_smiles = fragments.canonical_smiles(smiles_list)
    pva = self.predictions
    predicted = list(pva['pred_avg'])  # List of predicted value for test molecules
    test_mol = list(pva['smiles'])  # List of test molecules
    r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
    feature_length = len(self.feature_list)  # Total amount of features
    return r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol


def nodes(self):
    """
    Objective: Create or merge Neo4j nodes from data collected from the ML pipeline
    Intent: While most of the nodes are merged, some need to be created instead because:
                - They don't need to be merged: MLModel
                - You can only merge Nodes on 1 main property key in py2neo. RandomSplit Nodes can have duplicate
                    properties with each other while still remain unique. For example: Splits can have the same test
                    percent, but not the same val percent. They can even have the same split percentage but not the same
                    random_seed. Therefore, RandomSplit nodes must be merged using Cypher instead of py2neo, which is
                    located in "rel_to_neo4j.py"
    Note: If you want to know why I put number of features in a list (line 77), read my note located in
                                                                                                "prep_from_output"
    """
    print("Creating Nodes for %s" % self.run_name)
    r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol = prep(self)

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
    feature_list = Node("FeatureList", name="FeatureList", num=[len(self.feature_list)])
    g.merge(feature_list, "FeatureList", "num")

    # Make TrainSet node
    train_set = Node("TrainSet", trainsize=self.n_train, name="TrainSet")
    g.merge(train_set, "TrainSet", "trainsize")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")

    # Since we can't use merge with multiple properties, I will merge RandomSplit nodes later on
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed,
                       val_percent=self.val_percent)
    g.create(randomsplit)

    # Make TestSet node
    testset = Node("TestSet", name="TestSet", RMSE=rmse, mse=mse, r2=r2, testsize=self.n_test)
    g.merge(testset, "TestSet", "RMSE")

    # Make ValidateSet node
    if self.val_percent > 0:
        valset = Node("ValidateSet", name="ValidateSet", valsize=self.n_val)
        g.merge(valset, "ValidateSet", "valsize")
    else:
        pass

    # Make nodes for all the SMILES
    for smiles, target in zip(canonical_smiles, list(self.target_array)):
        record = g.run("""MATCH (n:SMILES {SMILES:"%s"}) RETURN n""" % smiles)
        if len(list(record)) > 0:
            print(f"This SMILES, {smiles}, already exist. Updating its properties and relationships")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = mol.measurement + $target, "
                       "mol.dataset = mol.dataset + $dataset",
                       parameters={'smiles': smiles, 'target': target, 'dataset': self.dataset})
        else:
            mol = Node("SMILES", SMILES=smiles)
            g.merge(mol, "SMILES", "SMILES")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = [$target],"
                       "mol.dataset = [$dataset]",
                       parameters={'smiles': smiles, 'target': target, 'dataset': self.dataset})

    # Only create Features for rdkit2d method
    df = self.data.loc[:, 'BalabanJ':'qed']
    columns = list(df.columns)

    # Create nodes and relationships between features, feature methods and SMILES
    for column in tqdm(columns, desc="Creating nodes and relationships between SMILES and features for rdkit2d"):
        # Create nodes for features
        features = Node(column, name=column)
        # Merge relationship
        g.merge(features, column, "name")
        g.evaluate("""match (rdkit2d:FeatureMethod {feature:"rdkit2d"}), (feat:%s) 
                        merge (rdkit2d)-[:CALCULATES]->(feat)""" % column)
        for mol, value in zip(canonical_smiles, list(df[column])):
            g.run("match (smile:SMILES {SMILES:$mol}), (feat:%s)"
                  "merge (smile)-[:HAS_DESCRIPTOR {value:$value}]->(feat)" % column,
                  parameters={'mol': mol, 'value': value})
