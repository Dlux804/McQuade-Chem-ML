"""
Objective: The goal of this script is to create nodes in Neo4j directly from the pipeline using class instances
"""

import pandas as pd
import time
from core.neo4j.make_query import Query
from core.storage.dictionary import target_name_grid
from py2neo import Graph, Node

from core.neo4j.fragments import smiles_to_frag, insert_fragments
from core.storage.misc import parallel_apply


# TODO REDO DOCSTRINGS


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
    actual = pva['actual']
    pva_predictions = pva.drop(['pred_avg', 'pred_std', 'smiles', 'actual'], axis=1)

    average_error = list(pva_predictions.sub(actual, axis=0).mean(axis=1))  # Calculate avg prediction error
    test_mol_dict = pd.DataFrame({'smiles': list(pva['smiles']), 'predicted': list(pva['pred_avg']),
                                  'uncertainty': list(pva['pred_std']),
                                  'error': average_error}).to_dict('records')
    return df_smiles, test_mol_dict, data_size


def __dataset_record__(self, graph):
    """
    Objective: Return record of dataset in the current database. If it's larger than 0, we can skip time and memory
                intensive functions
    :param self:
    :param graph
    :return:
    """

    record = graph.run("""MATCH (n:DataSet {data:"%s"}) RETURN n""" % self.dataset)
    return len(list(record))


def __make_molecules__(dict_records, dataset, graph):
    """
    Objective: Create SMILES nodes and their properties such as dataset and target value
    :param dict_records: Dictionary records that contain all SMILES and target value
    :param dataset: The dataset that the SMILES belong to
    :param graph: Connecting python to Neo4j
    :return:
    """
    t1 = time.perf_counter()
    molecule_query = Query(graph=graph).__make_molecules_query__(target_name=target_name_grid(dataset))
    graph.evaluate(molecule_query, parameters={'molecules': dict_records, 'dataset': dataset})
    t2 = time.perf_counter()
    print(f"Finished creating molecules in {t2 - t1}sec")


def __merge_molecules_and_rdkit2d__(df, graph):
    """
    Objective: For every row in a csv (or dataframe) that contains SMILES and rdkit2d Features, merge SMILES with
    rdkit2d features with its respective feature values
    Intent: Created to be used with dataframe's apply function.
    :param df:
    :param graph:
        """

    mol_rdkit2d_query = Query(graph=graph).__molecules_and_rdkit2d_query__()

    # TODO make this work without a for loop if possible
    molecules = []
    for index, row in df.iterrows():
        row_feats = []
        row = dict(row)
        smiles = row.pop('smiles')
        for feat_name, feat_value in row.items():
            row_feats.append({'name': feat_name, 'value': feat_value})
        molecules.append({'smiles': smiles, 'feats': row_feats})

    # APOC is slower in this part than using UNWIND alone
    range_molecules = []
    for index, molecule in enumerate(molecules):
        range_molecules.append(molecule)
        if index % 2000 == 0 and index != 0:
            tx = graph.begin(autocommit=True)
            tx.evaluate(mol_rdkit2d_query, parameters={"molecules": range_molecules})
            range_molecules = []
    if range_molecules:
        tx = graph.begin(autocommit=True)
        tx.evaluate(mol_rdkit2d_query, parameters={"molecules": range_molecules})


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
    self.tuned = str(self.tuned).capitalize()
    t1 = time.perf_counter()
    self.tuned = str(self.tuned).capitalize()
    print("Creating Nodes for %s" % self.run_name)

    g = Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Define graph for function

    df_smiles, test_mol_dict, data_size = prep(self)

    query = Query(graph=g)
    query.__check_for_constraints__()

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
                 test_time=float(self.predictions_stats["time_avg"]), tasktype=self.task_type)
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

    # Check to see if we have created fragments and molecules for this run's dataset
    record = __dataset_record__(self, graph=g)
    if record > 0:  # If found dataset
        print(f"This dataset, {self.dataset} already exists in the database. Skipping fragments, and rdkit2d features")
    else:  # If unique dataset
        # Make molecules
        __make_molecules__(dict_records=df_smiles.to_dict('records'), dataset=self.dataset, graph=g)
        t3 = time.perf_counter()
        df_for_fragments = df_smiles.drop(['target'], axis=1)
        print('Calculating Fragments')
        df_for_fragments['fragments'] = parallel_apply(df_for_fragments['smiles'], smiles_to_frag, number_of_workers=3
                                                       , loading_bars=False)
        print('Inserting Fragments')
        insert_fragments(df_for_fragments, graph=g)  # Make fragments
        t4 = time.perf_counter()
        print(f"Finished creating molecular fragments in {t4 - t3}sec")

        if 'rdkit2d' in self.feat_method_name:  # rdkit2d in feat method name
            t5 = time.perf_counter()
            print("Creating rdki2d features")
            df_rdkit2d_features = self.data.filter(regex='smiles|fr_|Count|Num|Charge|TPSA|qed|%s' % self.target_name,
                                                   axis=1)
            __merge_molecules_and_rdkit2d__(df_rdkit2d_features, graph=g)  # Make rdkit2d
            t6 = time.perf_counter()
            print(f"Finished creating nodes and relationships between SMILES and rdkit2d features in {t6 - t5}sec")
        else:
            pass

    # Make FeatureMethod node
    for feat in self.feat_method_name:
        feature_method = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_method, "FeatureMethod", "feature")

    # Make dataset node
    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset,
                   measurement=target_name_grid(self.dataset), tasktype=self.task_type, datasize=data_size)
    g.merge(dataset, "DataSet", "data")
    t7 = time.perf_counter()
    print(f"Finished creating nodes in {t7-t1}sec")
