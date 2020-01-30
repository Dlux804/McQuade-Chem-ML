import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


def insert_ml_runs(file):
    graph = Graph()
    ml_data = pd.read_csv(file)
    ml_dicts = ml_data.to_dict()
    for i in range(len(ml_dicts)):
        ml_dict = ml_dicts[i]
        ml_model = Node("model", algorithm=ml_dict['algorithm'], data=ml_dict['dataset'])
        graph.merge(ml_model, 'model', 'algorithm')

insert_ml_runs('merged_MLoutput.csv')