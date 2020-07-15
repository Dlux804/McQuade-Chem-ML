"""
Objective: Gather data from _data.csv and _attributes.csv files and import nodes based on our ontology to Neo4j
"""
import pandas as pd
from py2neo import Graph, Node
from tqdm import tqdm
from core.fragments import fragments_to_neo
import time

# TODO REDO DOCSTRING

# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def __merge_molecules_and_rdkit2d__(row):
    """
    Objective: For every row in a csv (or dataframe) that contains SMILES and rdkit2d Features, merge SMILES with
    rdkit2d features with its respective feature values
    Intent: Created to be used with dataframe's apply function.
    :param row: A row in a csv
    :return:
        """
    mol_feat_query = """
    UNWIND $molecule as molecule
    MATCH (rdkit2d:FeatureMethod {feature:"rdkit2d"})
    MERGE (mol:Molecule {SMILES: molecule.smiles})
        FOREACH (feat in molecule.feats|
            MERGE (feature:Feature {name: feat.name})
            MERGE (mol)-[:HAS_DESCRIPTOR {value: feat.value, feat_name:feat.name}]->(feature)
                )
    MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
    """

    row = dict(row)
    smiles = row.pop('smiles')

    feats = pd.DataFrame({'name': list(row.keys()), 'value': list(row.values())}).to_dict('records')

    molecule = {'smiles': smiles, 'feats': feats}
    tx = g.begin(autocommit=True)
    tx.evaluate(mol_feat_query, parameters={"molecule": molecule})


def nodes(prep):
    """
    Objective: Create or merge Neo4j nodes from data collected from the ML pipeline
    Intent: While most of the nodes are merged, some need to be created instead because:
                - They don't need to be merged: MLModel
                - You can only merge Nodes on 1 main property key in py2neo. RandomSplit Nodes and others
                  can have duplicate properties with each other while still remain unique. For example: Splits can have
                  the same test percent, but not the same val percent. They can even have the same split percentage but
                  not the same random_seed. Therefore, RandomSplit nodes must be merged using Cypher instead of py2neo
                  in "rel_to_neo4j.py". Same with TrainSet and Valset
    Note: If you want to know why I put number of features in a list (line 77), read my note located in
                                                                                                "prep_from_output"
    """
    t1 = time.perf_counter()
    print("Creating Nodes for %s" % prep.run_name)
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
    train_set = Node("TrainSet", trainsize=prep.n_train, name="TrainSet", random_seed=prep.random_seed)
    g.create(train_set)

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
        valset = Node("ValidateSet", name="ValidateSet", valsize=prep.n_val, random_seed=prep.random_seed)
        g.create(valset)
    else:
        pass
    
    # Create SMILES
    prep.df_smiles.columns = ['smiles', 'target']  # Change target column header to target
    g.evaluate("""
        UNWIND $molecules as molecule
        MATCH (mol:Molecule {SMILES: molecule.smiles})
        SET mol.target = [molecule.target], mol.dataset = [$dataset]
        """, parameters={'molecules': prep.df_smiles.to_dict('records'), 'dataset': prep.dataset_str})
    t2 = time.perf_counter()
    print(f"Finished creating main ML nodes in {t2 - t1}sec")

    # Creating nodes and relationships between SMILES, Features
    if ["rdkit2d"] in prep.feat_method_name:
        record = g.run("""MATCH (n:DataSet {data:"%s"}) RETURN n""" % prep.dataset_str)
        if len(list(record)) > 0:
            print(f"This dataset, {prep.dataset_str},and its molecular features already exist in the database. Moving on")
        else:
            df_rdkit2d_features = prep.rdkit2d_features.drop(['target'], axis=1)
            df_rdkit2d_features.apply(__merge_molecules_and_rdkit2d__, axis=1)
            t3 = time.perf_counter()
            print(f"Finished creating nodes and relationships between SMILES and their features in {t3 - t2}sec")
            prep.df_from_data[['smiles']].apply(fragments_to_neo, axis=1)
            t4 = time.perf_counter()
            print(f"Finished creating molecular fragments in {t4 - t3}sec")

    else:
        g.evaluate("""
                 UNWIND $feat_name as featname
                 match (method:FeatureMethod {feature: featname}), (featlist:FeatureList {num:$feat_num})
                 merge (featurelist)<-[:CONTRIBUTES_TO]-(method)
                """, parameters={'feat_name': prep.feat_method_name, 'feat_num': [len(prep.num_feature_list)]})

        # Make dataset node
        dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=prep.dataset_str, measurement=prep.target_name)
        g.merge(dataset, "DataSet", "data")