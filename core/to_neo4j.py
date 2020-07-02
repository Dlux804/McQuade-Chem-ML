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


merge = """MATCH (n)
WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count
WHERE count > 1
CALL apoc.refactor.mergeNodes(nodelist) YIELD node
RETURN node

"""




def algorithm(self):
    """"""
    algor = ogm_class.Algorithm()
    if self.algorithm == "nn":
        algor.source = "Keras"
    else:
        algor.source = "sklearn"
    algor.tuned = self.tuned
    algor.name = self.algorithm
    g.push(algor)


def features(self):
    """

    :param self:
    :return:
    """
    for feat in self.feat_name:
        feature_name = Node(feat, feature=feat, name=feat)
        g.create(feature_name)


def to_neo4j(self):
    """"""
    tuning_algorithm = Node("TuningAlg", name="TuningAlg", algorithm="BayesianOptimizer", num_cv=self.cv_folds,
                            tuneTime=self.tune_time, delta=self.cp_delta, n_best=self.cp_n_best, steps=self.opt_iter)
    g.create(tuning_algorithm)

    model = Node("MLModel", name=self.run_name, feat_time=self.feat_time, date=self.date, train_time=self.tune_time,
                 test_time=self.predictions_stats["time_avg"])
    g.create(model)

    feature_list = Node("FeatureList", name="FeatureList", num=len(self.feature_list))
    g.create(feature_list)

    trainsing_size = len(self.target_array) * self.train_percent
    training_set = Node("TrainingSet", size=trainsing_size, name="TrainingSet")
    g.create(training_set)

    dataset = Node("DataSet", name="Dataset", source="Moleculenet", data=self.dataset, measurement=self.target_name)
    g.create(dataset)

    randomsplit = Node("RamdomSplit", name="RandomSplit", test_percent=self.test_percent,
                       train_percent=self.train_percent, random_seed=self.random_seed)
    g.create(randomsplit)

    pva = self.predictions
    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))

    testset = Node("TestSet", name="TestSet", RMSE=rmse, mse=mse, r2=r2)



    # g.evaluate(merge)



# to_neo4j()