"""
Objective: Gather data from _data.csv and _attributes.csv files and import nodes based on our ontology to Neo4j
"""

from py2neo import Graph, Node
from tqdm import tqdm
from core import fragments


# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def nodes(prep):
    """
    Objective: Create Neo4j nodes from output data. Merge them if they already exist
    Intent: I want this file to almost exclusive create nodes in Neo4j. The only exception is that it also create
            relationships between SMILES and rdkit2d features if the run uses rdkit2d.
    :param prep: A class variable that contains all necessary data from "output" files. The file that creates this class
                variable is "prep_from_outputs.py"
    :return:
    """
    print("Creating Nodes for %s" % prep.run_name)
    # r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol = prep(self)

    # Make algorithm node

    if prep.algorithm == "nn":
        algor = Node("Algorithm", name=prep.algorithm, source="Keras", tuned=prep.tuned)
        g.merge(algor, "Algorithm", "name")
    else:
        algor = Node("Algorithm", name=prep.algorithm, source="sklearn", tuned=prep.tuned)
        g.merge(algor, "Algorithm", "name")

    # Make FeatureMethod node
    for feat in prep.feat_method_name:
        feature_method = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_method, "FeatureMethod", "feature")

    # Make Tuner node
    if prep.tuned:
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=prep.cv_folds)
        g.merge(tuning_algorithm, "TuningAlg", "algorithm")
    else:
        tuning_algorithm = Node("NotTuned", name="NotTuned")
        g.merge(tuning_algorithm, "NotTuned", "name")

    # Make MLModel nodes
    model = Node("MLModel", name=prep.run_name, feat_time=prep.feat_time, date=prep.date, train_time=prep.tune_time,
                 test_time=prep.test_time)
    g.create(model)

    # Make FeatureList node
    feature_list = Node("FeatureList", name="FeatureList", num=prep.num_feature_list)
    g.merge(feature_list, "FeatureList", "num")

    # Make TrainSet node
    train_set = Node("TrainSet", trainsize=prep.n_train, name="TrainSet")
    g.merge(train_set, "TrainSet", "trainsize")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=prep.dataset_str, measurement=prep.target_name)
    g.merge(dataset, "DataSet", "data")

    # Since we can't use merge with multiple properties, I will merge RandomSplit nodes later on
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=prep.test_percent,
                       train_percent=prep.train_percent, random_seed=prep.random_seed,
                       val_percent=prep.val_percent)
    g.create(randomsplit)

    # Make TestSet node

    testset = Node("TestSet", name="TestSet", RMSE=prep.rmse, mse=prep.mse, r2=prep.r2, testsize=prep.n_test)
    g.merge(testset, "TestSet", "RMSE")
    #
    # Make ValidateSet node
    if prep.val_percent > 0:
        valset = Node("ValidateSet", name="ValidateSet", valsize=prep.n_val)
        g.merge(valset, "ValidateSet", "valsize")
    else:
        pass

    # Make nodes for all the SMILES
    for smiles, target in zip(prep.canonical_smiles, prep.target_array):
        record = g.run("""MATCH (n:SMILES {SMILES:"%s"}) RETURN n""" % smiles)  # See if
        if len(list(record)) > 0:
            print(f"This SMILES, {smiles}, already exists in your graph db. Updating its properties and relationships")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = mol.measurement + $target, "
                       "mol.dataset = mol.dataset + $dataset",
                       parameters={'smiles': smiles, 'target': target, 'dataset': prep.dataset_str})
        else:
            mol = Node("SMILES", SMILES=smiles)
            g.merge(mol, "SMILES", "SMILES")
            g.evaluate("match (mol:SMILES {SMILES: $smiles}) set mol.measurement = [$target],"
                       "mol.dataset = [$dataset]",
                       parameters={'smiles': smiles, 'target': target, 'dataset': prep.dataset_str})

    # # Only create Features for rdkit2d method
    # Create nodes and relationships between features, feature methods and SMILES
    if ['rdkit2d'] in prep.feat_method_name:
        print("This run has rdkit2d")
        for column in tqdm(prep.features_col, desc="Creating nodes and rel between SMILES and features for rdkit2d"):
            # Create nodes for features
            features = Node(column, name=column)
            # Merge relationship
            g.merge(features, column, "name")
            g.evaluate("""match (rdkit2d:FeatureMethod {feature:"rdkit2d"}), (feat:%s)
                          merge (rdkit2d)-[:CALCULATES]->(feat)""" % column)
            for mol, value in zip(prep.canonical_smiles, list(prep.rdkit2d_features[column])):
                g.run("match (smile:SMILES {SMILES:$mol}), (feat:%s)"
                      "merge (smile)-[:HAS_DESCRIPTOR {value:$value}]->(feat)" % column,
                      parameters={'mol': mol, 'value': value})
    else:
        pass

    # Create molecular fragments for smiles
    fragments.fragments_to_neo(prep.canonical_smiles)