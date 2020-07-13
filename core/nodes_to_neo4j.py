"""
Objective: The goal of this script is to create nodes in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from core.fragments import fragments_to_neo
from core import fragments
import pandas as pd
import time
# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def prep(self):
    """
    Objective: Calculate or prepare variables that we don't have class instances for
    Intent: I want to have one function that calculates the data I need for the ontology. I don't think we need
            class instances for these data since they can be easily obtained with one ot two lines of code each.
    """
    canonical_smiles = fragments.canonical_smiles(list(self.data['smiles']))
    pva = self.predictions
    predicted = list(pva['pred_avg'])  # List of predicted value for test molecules
    test_mol = list(pva['smiles'])  # List of test molecules
    r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
    feature_length = len(self.feature_list)  # Total amount of features
    df_smiles = self.data.iloc[:, [1, 2]]
    df_features = self.data.loc[:, 'smiles':'qed']
    return r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol, df_smiles, df_features


def __merge_molecules_and_feats__(row):

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
    t1 = time.perf_counter()
    print("Creating Nodes for %s" % self.run_name)

    r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol, df_smiles, df_features = prep(self)

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
    df_smiles.columns = ['smiles', 'target']
    # Create SMILES
    g.evaluate("""
    UNWIND $molecules as molecule
    MERGE (mol:Molecule {SMILES: molecule.smiles})
    SET mol.target = [molecule.target], mol.dataset = [$dataset]
    """, parameters={'molecules': df_smiles.to_dict('records'), 'dataset': self.dataset})
    t2 = time.perf_counter()
    print(f"Finished creating main ML nodes in {t2-t1}sec")

    # Creating nodes and relationships between SMILES, Features
    if "rdkit2d" in self.feat_method_name:
        record = g.run("""MATCH (n:DataSet {data:"%s"}) RETURN n""" % self.dataset)
        if len(list(record)) > 0:
            print(f"This dataset, {self.dataset},and its molecular features already exist in the database. Moving on")
        else:
            df_features = df_features.drop([self.target_name], axis=1)
            df_features.apply(__merge_molecules_and_feats__, axis=1)
            t3 = time.perf_counter()
            print(f"Finished creating nodes and relationships between SMILES and their features in {t3 - t2}sec")
            self.data[['smiles']].apply(fragments_to_neo, axis=1)
            t4 = time.perf_counter()
            print(f"Finished creating molecular fragments in {t4 - t3}sec")

    else:
        pass

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")

