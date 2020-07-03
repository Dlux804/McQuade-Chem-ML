"""
Objective: The goal of this script is - by working in conjunction with ogm_class.py - create graphs in Neo4j using model
instance

"""

from core import ogm_class, models, misc
from py2neo import Graph, Relationship, Node
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


# merge = """MATCH (n)
# WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count
# WHERE count > 1
# CALL apoc.refactor.mergeNodes(nodelist) YIELD node
# RETURN node
#
# """




def calculate(self):
    """"""
    training_size = len(self.target_array) * self.train_percent
    test_size = len(self.target_array) * self.test_percent
    pva = self.predictions
    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))
    feature_length = len(self.feature_list)
    return training_size, test_size, pva, r2, mse, rmse, feature_length

def nodes(self):
    """"""
    training_size, test_size, pva, r2, mse, rmse, feature_length = calculate(self)

    if self.algorithm == "nn":
        algor = Node("Algorithm", name=self.algorithm, source="Keras", tuned=self.tuned)
        g.create(algor)
    else:
        algor = Node("Algorithm", name=self.algorithm, source="sklearn", tuned=self.tuned)
        g.create(algor)

    for feat in self.feat_name:
        feature_name = Node(feat, feature=feat, name=feat)
        g.create(feature_name)

    if self.tuned:
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
                            tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best, steps=self.opt_iter)
        g.create(tuning_algorithm)
    else:
        self.cp_delta = "Not tuned"
        self.cp_n_best = "Not tuned"
        tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
                                tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best,
                                steps=self.opt_iter)
        g.create(tuning_algorithm)


    model = Node("MLModel", name=self.run_name, feat_time=self.feat_time, date=self.date, train_time=self.tune_time,
                 test_time=self.predictions_stats["time_avg"])
    g.create(model)

    feature_list = Node("FeatureList", name="FeatureList", num=len(self.feature_list))
    g.create(feature_list)

    training_set = Node("TrainingSet", trainsize=training_size, name="TrainingSet")
    g.create(training_set)

    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.create(dataset)
    randomsplit = Node("RandomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed,
                       val_percent=self.val_percent)
    g.create(randomsplit)

    testset = Node("TestingSet", name="TestingSet", RMSE=rmse, mse=mse, r2=r2, testsize=test_size)
    g.create(testset)

    smiles_list = list(self.data['smiles'])
    for smi, target in zip(smiles_list, list(self.target_array)):
        mol = Node("SMILES", SMILES=smi, measurement=target)
        g.create(mol)
    # g.evaluate(merge)


def relationships(self):
    training_size, test_size, pva, r2, mse, rmse, feature_length = calculate(self)

    query_line = """ 
    match (testset:TestingSet {testsize:%f})
    
    create (model)-[:USES_ALGORITHM]->(algorithms)-[:USES_TUNING]->(tuner), (dataset)<-[:USES_DATASET]-(model)""" \
                 % (test_size)
    g.evaluate(query_line)

# to_neo4j()

    # match (model:MLModel), (Tuner:TuningAlg), (algorithms:Algorithm {name:"%s"}), (dataset:Dataset {data:"%s"}),
    # (featurelist:FeatureList {num:%d}), (trainset:TrainingSet {size:%f}), (testset:TestSet {size:%f}),
    # (randomsplit:RandomSplit {test_percent:%f, train_percent:%f})
    # create (dataset)-[:USES_DATASET]->(model)-[:USES_ALGORITHM]->(algorithms) """ \
    #              % (self.algorithm, self.dataset, feature_length, trainsing_size, test_size, self.test_percent,
    #                 self.train_percent)