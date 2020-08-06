"""
Objective: The goal of this script is to create nodes in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from core.neo4j.fragments import fragments_to_neo
from core.neo4j import fragments
import pandas as pd
import time
from core.neo4j.make_query import Query
from core.storage.dictionary import target_name_grid
# TODO Add docstring


def prep(self):
    """
    Objective: Calculate or prepare variables that we don't have class instances for
    Intent: I want to have one function that calculates the data I need for the ontology. I don't think we need
            class instances for these data since they can be easily obtained with one ot two lines of code each.
    """
    pva = self.predictions
    data_size = len(self.data['smiles'])
    # self.data['smiles'] = canonical_smiles
    df_smiles = self.data.iloc[:, [1, 2]]

    test_mol_dict = pd.DataFrame({'smiles': list(pva['smiles']), 'predicted': list(pva['pred_avg']),
                                  'uncertainty': list(pva['pred_std'])}).to_dict('records')
    return df_smiles, test_mol_dict, data_size


def __dataset_record__(self, g):
    """
    Objective: Return record of dataset in the current database. If it's larger than 0, we can skip time and memory
                intensive functions
    :param self:
    :return:
    """

    record = g.run("""MATCH (n:DataSet {data:"%s"}) RETURN n""" % self.dataset)
    return len(list(record))


def __make_molecules__(df_records, dataset, size, g):
    """
    Objective: Create SMILES nodes and
    :param df_records:
    :param dataset:
    :param target_name:
    :param size:
    :param g:
    :return:
    """
    t1 = time.perf_counter()
    molecule_query = Query(size=size).__make_molecules_query__(target_name=target_name_grid(dataset))
    g.evaluate(molecule_query, parameters={'molecules': df_records, 'dataset': dataset})
    t2 = time.perf_counter()
    print(f"Finished creating molecules in {t2 - t1}sec")


def __merge_molecules_and_rdkit2d__(row, size, g):
    """
    Objective: For every row in a csv (or dataframe) that contains SMILES and rdkit2d Features, merge SMILES with
    rdkit2d features with its respective feature values
    Intent: Created to be used with dataframe's apply function.
    :param row: A row in a csv
    :return:
        """

    mol_rdkit2d_query = Query(size=size).__molecules_and_rdkit2d_query__()
    row = dict(row)
    smiles = row.pop('smiles')

    feats = pd.DataFrame({'name': list(row.keys()), 'value': list(row.values())}).to_dict('records')

    molecule = {'smiles': smiles, 'feats': feats}
    tx = g.begin(autocommit=True)
    tx.evaluate(mol_rdkit2d_query, parameters={"molecule": molecule})


def nodes(self, from_output=False):
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

    g = Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Define graph for function

    df_smiles, test_mol_dict, data_size = prep(self)

    query = Query(size=data_size)
    query.__check_for_constraints__(g)

    # Make algorithm node
    if self.algorithm == "nn":
        algor = Node("Algorithm", name=self.algorithm, source="Keras", tuned=self.tuned)
        g.create(algor)
        g.evaluate(""" MATCH (n:Algorithm) WITH n.name AS name, n.tuned as tuned,
                       COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
                       CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")
    else:
        algor = Node("Algorithm", name=self.algorithm, source="sklearn", tuned=self.tuned)
        g.create(algor)
        g.evaluate(""" MATCH (n:Algorithm) WITH n.name AS name, n.tuned as tuned,
                       COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
                       CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")

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
    feat_list = Node("FeatureList", name="FeatureList", num=self.feature_length, feat_ID=self.feat_meth,
                     featuure_lists=str(self.feature_list))
    g.merge(feat_list, "FeatureList", "feat_ID")

    # Make TrainSet node
    train_set = Node("TrainSet", trainsize=self.n_train, name="TrainSet", random_seed=self.random_seed)
    g.create(train_set)

    # Since we can't use merge with multiple properties, I will merge RandomSplit nodes later on
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed,
                       val_percent=self.val_percent)
    g.create(randomsplit)

    # Make TestSet node
    testset = Node("TestSet", name="TestSet", testsize=self.n_test, random_seed=self.random_seed)
    g.create(testset)

    # Make ValidateSet node
    if self.val_percent > 0:
        valset = Node("ValSet", name="ValidateSet", valsize=self.n_val, random_seed=self.random_seed)
        g.create(valset)
    else:
        pass

    # Create SMILES
    # SET FUNCTION WILL UPDATE DUPLICATE VALUES TO LIST
    df_smiles.columns = ['smiles', 'target']  # Change target column header to target
    # Creating Molecular Fragments
    # Check to see if we have created fragments for this run's dataset
    record = __dataset_record__(self, g=g)
    if record > 0:
        print(f"This dataset, {self.dataset} already exists in the database. Skipping fragments, and rdkit2d features")
    else:  # If unique dataset
        # Make molecules
        __make_molecules__(df_records=df_smiles.to_dict('records'), dataset=self.dataset, size=data_size, g=g)
        t3 = time.perf_counter()
        self.data[['smiles']].apply(fragments_to_neo, size=data_size, g=g, axis=1)  # Create molecular fragments
        t4 = time.perf_counter()
        print(f"Finished creating molecular fragments in {t4 - t3}sec")

        if 'rdkit2d' in self.feat_method_name:  # rdkit2d in feat method name
            t5 = time.perf_counter()
            print("Creating rdki2d features")
            df_rdkit2d_features = self.data.filter(regex='smiles|fr_|Count|Num|Charge|TPSA|qed|%s' % self.target_name,
                                                   axis=1)
            df_rdkit2d_features.apply(__merge_molecules_and_rdkit2d__, size=data_size, g=g, axis=1)  # Make rdkit2d
            t6 = time.perf_counter()
            print(f"Finished creating nodes and relationships between SMILES and rdkit2d features in {t6 - t5}sec")
        else:
            pass

    # Make FeatureMethod node
    for feat in self.feat_method_name:
        feature_method = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_method, "FeatureMethod", "feature")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")
    t7 = time.perf_counter()
    print(f"Finished creating nodes in {t7-t1}sec")