"""
Objective: The goal of this script is - by working in conjunction with ogm_class.py - create graphs in Neo4j using model
instance

"""

from core import ogm_class, models, misc
from py2neo import Graph, Relationship, Node
import pathlib, os
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


def model(self):
    """"""
    model = ogm_class.MlModel()
    model.name = self.run_name
    model.feat_time = self.feat_time
    model.date = self.date
    model.train_time = self.tune_time





    # g.evaluate(merge)



# to_neo4j()