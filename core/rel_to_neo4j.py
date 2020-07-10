"""
Objective: The goal of this script is to create relationships in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph
from core import nodes_to_neo4j


# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def relationships(self):
    """
    Objective: Create relationships in Neo4j based on our ontology directly from the pipeline.
    Intent: I want this script to only create relationships for the nodes in Neo4j. There are also some node merging
            lines sprinkled in. Sometimes when you merge two nodes, their duplicate relationships don't merge with each
            other.
    """
    print("Creating relationships...")
    r2, mse, rmse, feature_length, canonical_smiles, predicted, test_mol = nodes_to_neo4j.prep(self)

    # Merge RandomSplit node
    g.evaluate(" MATCH (n:RandomSplit) WITH n.test_percent AS test, n.train_percent as train, n.random_seed as seed,"
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

    # MLModel to Algorithm
    if self.tuned:  # If tuned
        param_dict = dict(self.params)  # Turn OrderedDict to dictionary
        for key in param_dict:
            g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
                       "merge (model)-[r:USES_ALGORITHM]->(algor) Set r.%s = $params" % key,
                       parameters={'algorithm': self.algorithm, 'run_name': self.run_name, 'params': param_dict[key]})
    else:  # If not tuned
        g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
                   "merge (model)-[r:USES_ALGORITHM]->(algor)",
                   parameters={'algorithm': self.algorithm, 'run_name': self.run_name})

    # Algorithm to TuningAlg
    if self.tuned:  # If tuned
        g.evaluate("match (tuning_alg:TuningAlg {tuneTime: $tune_time}), (algor:Algorithm {name: $algorithm})"
                   "merge (algor)-[:USES_TUNING]->(tuning_alg)",
                   parameters={'tune_time': self.tune_time, 'algorithm': self.algorithm})
    else:
        pass  # not tuned
    # MLModel to DataSet
    g.evaluate("match (dataset:DataSet {data: $dataset}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_DATASET]->(dataset)",
               parameters={'dataset': self.dataset, 'run_name': self.run_name})

    # MLModel to Featurelist
    g.evaluate("match (featurelist:FeatureList {num: $feat_length}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_FEATURES]->(featurelist)",
               parameters={'feat_length': feature_length, 'run_name': self.run_name})

    # MLModel to RandomSplit
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, "
               "random_seed: $random_seed}), "
               "(model:MLModel {name: $run_name})"
               "merge (model)-[:USES_SPLIT]->(split)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'run_name': self.run_name, 'random_seed': self.random_seed})

    # MLModel to TrainingSet
    g.evaluate("match (trainset:TrainSet {trainsize: $training_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:TRAINS]->(trainset)",
               parameters={'training_size': self.n_train, 'run_name': self.run_name})

    # MLModel to TestSet
    g.evaluate("match (testset:TestSet {testsize: $test_size, RMSE: $rmse}), (model:MLModel {name: $run_name})"
               "merge (model)-[:PREDICTS]->(testset)",
               parameters={'test_size': self.n_test, 'rmse': rmse, 'run_name': self.run_name})

    # MLModel to ValidateSet
    g.evaluate("match (validate:ValidateSet {valsize: $val_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:VALIDATE]->(validate)",
               parameters={'val_size': self.n_val, 'run_name': self.run_name})

    # MLModel to feature method, FeatureList to feature method
    for feat in self.feat_method_name:
        g.evaluate("match (method:FeatureMethod {feature: $feat}), (model:MLModel {name: $run_name}) "
                   "merge (model)-[:USES_FEATURIZATION]->(method)",
                   parameters={'feat': feat, 'run_name': self.run_name})
        g.evaluate("match (method:FeatureMethod {feature: $feat}), (featurelist:FeatureList {num: $feat_length}) "
                   "merge (method)-[:CONTRIBUTES_TO]->(featurelist)",
                   parameters={'feat': feat, 'feat_length': feature_length})

    # RandomSplit to TrainSet and TestSet
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}),"
               "(testset:TestSet {testsize: $test_size, RMSE: $rmse}), "
               "(trainset:TrainSet {trainsize: $training_size})"
               "merge (trainset)<-[:SPLITS_INTO]-(split)-[:SPLITS_INTO]->(testset)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'test_size': self.n_test, 'rmse': rmse, 'training_size': self.n_train})
    # RandomSplit to Validation
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, "
               "random_seed: $random_seed}), "
               "(validate:ValidateSet {valsize: $val_size}) merge (split)-[:SPLITS_INTO]->(validate)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'val_size': self.n_val, 'random_seed': self.random_seed})

    # Connect TrainSet with its molecules
    for train_smiles in list(self.train_molecules):
        g.evaluate("match (smile:SMILES {SMILES:$mol}), (trainset:TrainSet {trainsize: $training_size})"
                   "merge (smile)-[:CONTAINS_MOLECULES]->(trainset)",
                   parameters={'mol': train_smiles, 'training_size': self.n_train})

    # Connect TestSet with its molecules
    for test_smiles, predict in zip(test_mol, predicted):
        g.evaluate("match (smiles:SMILES {SMILES:$mol}), (testset:TestSet {testsize: $test_size, RMSE: $rmse})"
                   "merge (testset)-[:CONTAINS_MOLECULES {predicted_value: $predicted}]->(smiles)",
                   parameters={'mol': test_smiles, 'test_size': self.n_test, 'rmse': rmse, 'predicted': predict})

    # Connect ValidateSet with its molecules
    if self.val_percent > 0:
        for val_smiles in list(self.val_molecules):
            g.evaluate("match (smile:SMILES {SMILES:$mol}), (validate:ValidateSet {valsize: $val_size})"
                       "merge (smile)-[:CONTAINS_MOLECULES]->(validate)",
                       parameters={'mol': val_smiles, 'val_size': self.n_val})
    else:
        pass

    # Merge "SPLITS_INTO" relationship between RandomSplit and TrainSet
    g.evaluate("""MATCH (:RandomSplit)-[r:SPLITS_INTO]-(:TrainSet {trainsize:%d})
                WITH r.name AS name, COLLECT(r) AS rell, COUNT(*) AS count
                WHERE count > 1
                CALL apoc.refactor.mergeRelationships(rell) YIELD rel
                RETURN rel""" % self.n_train)