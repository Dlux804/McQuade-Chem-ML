"""
Objective: The goal of this script is - by working in conjunction with ogm_class.py - create graphs in Neo4j using model
instance

"""

from core import ogm_class, models, misc
from py2neo import Graph, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def prep(self):
    """"""
    smiles_list = list(self.data['smiles'])
    val_size = len(self.target_array) * self.val_percent
    training_size = len(self.target_array) * self.train_percent
    test_size = len(self.target_array) * self.test_percent
    pva = self.predictions
    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))
    feature_length = len(self.feature_list)
    return training_size, test_size, pva, r2, mse, rmse, feature_length, val_size, smiles_list


def nodes(self):
    """"""
    training_size, test_size, pva, r2, mse, rmse, feature_length, val_size, smiles_list = prep(self)
    # Make algorithm node
    if self.algorithm == "nn":
        algor = Node("Algorithm", name=self.algorithm, source="Keras", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")
    else:
        algor = Node("Algorithm", name=self.algorithm, source="sklearn", tuned=self.tuned)
        g.merge(algor, "Algorithm", "name")

    # Make FeatureMethod node
    for feat in self.feat_name:
        feature_name = Node("FeatureMethod", feature=feat, name=feat)
        g.merge(feature_name, "FeatureMethod", "feature")

    # Make Tuner node
    if self.tuned:
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
                            tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best, steps=self.opt_iter)
        g.merge(tuning_algorithm, "TuningAlg", "algorithm")
    else:
        pass  # If not tuned

        # self.cp_delta = "Not tuned"
        # self.cp_n_best = "Not tuned"
        # tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
        #                         tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best,
        #                         steps=self.opt_iter)
        # g.create(tuning_algorithm)

    model = Node("MLModel", name=self.run_name, feat_time=self.feat_time, date=self.date, train_time=self.tune_time,
                 test_time=self.predictions_stats["time_avg"])
    g.create(model)

    feature_list = Node("FeatureList", name="FeatureList", num=len(self.feature_list))
    g.merge(feature_list, "FeatureList", "num")

    training_set = Node("TrainSet", trainsize=training_size, name="TrainSet")
    g.merge(training_set, "TrainSet",  "trainsize")

    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.merge(dataset, "DataSet", "data")

    # Since we can't use merge with multiple properties, I will merge RandomSplit nodes later on
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed,
                       val_percent=self.val_percent)
    g.create(randomsplit)

    testset = Node("TestSet", name="TestSet", RMSE=rmse, mse=mse, r2=r2, testsize=test_size)
    g.merge(testset, "TestSet", "RMSE")

    valset = Node("ValidateSet", name="ValidateSet", valsize=val_size)
    g.merge(valset, "ValidateSet", "valsize")

    for smi, target in zip(smiles_list, list(self.target_array)):
        mol = Node("SMILES", SMILES=smi, measurement=target)
        g.merge(mol, "SMILES", "SMILES")
    # g.evaluate(merge)

def relationships(self):
    training_size, test_size, pva, r2, mse, rmse, feature_length, val_size, smiles_list = prep(self)

    # Merge RandomSplit node
    g.evaluate(" MATCH (n:RandomSplit) WITH n.test_percent AS test and n.train_percent as train, "
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")
#
# """")

    # # MLModel to Algorithm
    g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_ALGORITHM]->(algor)",
               parameters={'algorithm': self.algorithm, 'run_name': self.run_name})

    # Algorithm to TuningAlg
    if self.tuned:
        g.evaluate("match (tuning_alg:TuningAlg {tuneTime: $tune_time}), (algor:Algorithm {name: $algorithm})"
                   "merge (algor)-[:USES_TUNING]->(tuning_alg)",
                   parameters={'tune_time': self.tune_time, 'algorithm': self.algorithm})
    else:
        pass
    # MLModel to DataSet
    g.evaluate("match (dataset:DataSet {data: $dataset}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_DATASET]->(dataset)",
               parameters={'dataset': self.dataset, 'run_name': self.run_name})

    # MLModel to Featurelist
    g.evaluate("match (featurelist:FeatureList {num: $feat_length}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_FEATURES]->(featurelist)",
               parameters={'feat_length': feature_length, 'run_name': self.run_name})

    # MLModel to RandomSplit
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}), "
               "(model:MLModel {name: $run_name})"
               "merge (model)-[:USES_SPLIT]->(split)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'run_name': self.run_name})

    # MLModel to TrainingSet
    g.evaluate("match (trainset:TrainSet {trainsize: $training_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:TRAINS]->(trainset)",
               parameters={'training_size': training_size, 'run_name': self.run_name})

    # MLModel to TestSet
    g.evaluate("match (testset:TestSet {testsize: $test_size, RMSE: $rmse}), (model:MLModel {name: $run_name})"
               "merge (model)-[:PREDICTS]->(testset)",
               parameters={'test_size': test_size, 'rmse': rmse, 'run_name': self.run_name})

    # MLModel to ValidateSet
    g.evaluate("match (validate:ValidateSet {valsize: $val_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:VALIDATE]->(validate)",
               parameters={'val_size': val_size, 'run_name': self.run_name})

    # MLModel to feature method, FeatureList to feature method
    for feat in self.feat_name:
        g.evaluate("match (feat_method:FeatureMethod {feature: $feat}), (model:MLModel {name: $run_name}) "
                   "merge (model)-[:USES_FEATURIZATION]->(feat_method)",
                   parameters={'feat': feat, 'run_name': self.run_name})
        g.evaluate("match (feat_method:FeatureMethod {feature: $feat}), (featurelist:FeatureList {num: $feat_length}) "
                   "merge (feat_method)-[:CONTRIBUTES_TO]->(featurelist)",
                   parameters={'feat': feat, 'feat_length': feature_length})

    # RandomSplit to TrainSet and TestSet
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}),"
               "(testset:TestSet {testsize: $test_size, RMSE: $rmse}), "
               "(trainset:TrainSet {trainsize: $training_size})"
               "merge (trainset)<-[:SPLITS_INTO]-(split)-[:SPLITS_INTO]->(testset)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'test_size': test_size, 'rmse': rmse, 'training_size': training_size})
    # RandomSplit to Validation
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}), "
               "(validate:ValidateSet {valsize: $val_size}) merge (split)-[:SPLITS_INTO]->(validate)",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'val_size': val_size})

    # Only create Features for rdkit2d method
    df = self.data.loc[:, 'BalabanJ':'qed']
    columns = list(df.columns)

    # Create nodes and relationships between features, feature methods and SMILES
    for column in columns:
        # Create nodes for features
        features = Node(column, name=column)
        g.merge(features, column, "name")
        # Merge relationship
        g.evaluate(""" match (rdkit2d:FeatureMethod {feature:"rdkit2d"}), (feat:%s) 
                   merge (rdkit2d)-[:CALCULATES]->(feat)""" % column)
        for mol, value in zip(smiles_list, list(df[column])):
            g.run("match (smile:SMILES {SMILES:$mol}), (feat:%s)"
                       "merge (smile)-[:HAS_DESCRIPTOR {value:$value}]->(feat)" % column,
                       parameters={'mol': mol, 'value': value})



