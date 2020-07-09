"""
Objectvie: Gather data from _data.csv and _attributes.csv files and import Features and their relationships with other
nodes based on our ontology to Neo4j
"""

from py2neo import Graph, Node

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def feature_method(df_from_attributes, df_from_data):
    data_list = list(df_from_attributes.T.to_dict().values())
    feat_method = list(data_list['feat_method_name'])
    for feat in feat_method:
        g.evaluate(""" merge (method:FeatureMethod {feature:$feat, name:$feat})
                    """)
