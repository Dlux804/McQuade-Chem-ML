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
g = Graph("bolt://localhost:11002", user="neo4j", password="1234")

# TODO REDO DOCSTRINGS


def prep(self):
    """
    Objective: Calculate or prepare variables that we don't have class instances for
    Intent: I want to have one function that calculates the data I need for the ontology. I don't think we need
            class instances for these data since they can be easily obtained with one ot two lines of code each.
    """
    canonical_smiles = fragments.canonical_smiles(list(self.data['smiles']))
    pva = self.predictions
    r2 = r2_score(pva['actual'], pva['pred_avg'])  # r2 values
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])  # mse values
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))  # rmse values
    self.data['smiles'] = canonical_smiles
    df_smiles = self.data.iloc[:, [1, 2]]

    test_mol_dict = pd.DataFrame({'smiles': list(pva['smiles']), 'predicted': list(pva['pred_avg']),
                                  'uncertainty': list(pva['pred_std'])}).to_dict('records')
    return r2, mse, rmse, canonical_smiles, df_smiles, test_mol_dict


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
    MERGE (rdkit2d:FeatureMethod {feature:"rdkit2d"})
    MERGE (mol:Molecule {SMILES: molecule.smiles})
        FOREACH (feat in molecule.feats|
            MERGE (feature:Feature {name: feat.name})
            MERGE (mol)-[:HAS_DESCRIPTOR {value: feat.value, feat_name:feat.name}]->(feature)
            MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
                )
    
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
                - You can only merge Nodes on 1 main property key in py2neo. RandomSplit Nodes and others
                  can have duplicate properties with each other while still remain unique. For example: Splits can have
                  the same test percent, but not the same val percent. They can even have the same split percentage but
                  not the same random_seed. Therefore, RandomSplit nodes must be merged using Cypher instead of py2neo
                  in "rel_to_neo4j.py". Same with TrainSet and Valset
    Note: If you want to know why I put number of features in a list (line 77), read my note located in
                                                                                                "prep_from_output"
    """
    t1 = time.perf_counter()
    print("Creating Nodes for %s" % self.run_name)

    r2, mse, rmse, canonical_smiles, df_smiles, test_mol_dict = prep(self)

    # Make algorithm node
    if self.algorithm == "nn":
        algor = Node("Algorithm", name=self.algorithm, source="Keras", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")
    else:
        algor = Node("Algorithm", name=self.algorithm, source="sklearn", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")
    
    # Make Tuner node
    if self.tuned:
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm=self.tune_algorithm_name)
        g.merge(tuning_algorithm, "TuningAlg", "algorithm")
    else:
        tuning_algorithm = Node("NotTuned", name="NotTuned")
        g.merge(tuning_algorithm, "NotTuned", "name")

    # Make MLModel nodes
    model = Node("MLModel", name=self.run_name, feat_time=self.feat_time, date=self.date, train_time=self.tune_time,
                 test_time=float(self.predictions_stats["time_avg"]))
    g.merge(model, "MLModel", "name")

    # Make FeatureList node
    feature_list = Node("FeatureList", name="FeatureList", num=self.feature_length, feat_ID=self.feat_meth)
    g.merge(feature_list, "FeatureList", "feat_ID")

    # Make TrainSet node
    train_set = Node("TrainSet", trainsize=self.n_train, name="TrainSet", random_seed=self.random_seed)
    g.create(train_set)

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
        valset = Node("ValSet", name="ValidateSet", valsize=self.n_val, random_seed=self.random_seed)
        g.create(valset)
    else:
        pass

    # Create SMILES
    df_smiles.columns = ['smiles', 'target']  # Change target column header to target
    g.evaluate("""
    UNWIND $molecules as molecule
    MERGE (mol:Molecule {SMILES: molecule.smiles, name: "Molecule"})
    SET mol.target = [molecule.target], mol.dataset = [$dataset]
    """, parameters={'molecules': df_smiles.to_dict('records'), 'dataset': self.dataset})
    t2 = time.perf_counter()
    print(f"Finished creating main ML nodes in {t2-t1}sec")

    # Creating nodes and relationships between SMILES, Features
    record = g.run("""MATCH (n:DataSet {data:"%s"}) RETURN n""" % self.dataset)
    if len(list(record)) > 0:
        print(f"This dataset, {self.dataset}, and its fragments already exist in the database. Moving on")
    else:
        t3 = time.perf_counter()
        self.data[['smiles']].apply(fragments_to_neo, axis=1)
        t4 = time.perf_counter()
        print(f"Finished creating molecular fragments in {t4 - t3}sec")

    if 'rdkit2d' in self.feat_method_name:
        record = g.run("""MATCH (n:FeatureMethod {feature:"rdkit2d"}) RETURN n""")
        if len(list(record)) > 0:
            print(f"This dataset, {self.dataset}, and its rdkit2d features already exist in the database. Moving on")
        else:
            print("Creating rdki2d features")
            df_rdkit2d_features = self.data.loc[:, 'smiles':'qed']
            try:
                df_rdkit2d_features = df_rdkit2d_features.rename(columns={self.target_name: 'target'})
            except KeyError:
                pass
            df_rdkit2d_features = df_rdkit2d_features.drop('target', axis=1)
            df_rdkit2d_features.apply(__merge_molecules_and_rdkit2d__, axis=1)
            t3 = time.perf_counter()
            print(f"Finished creating nodes and relationships between SMILES and rdkit2d features in {t3 - t2}sec")

    # Make FeatureMethod node
    for feat in self.feat_method_name:
        print(feat)
        feature_method = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_method, "FeatureMethod", "feature")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")
