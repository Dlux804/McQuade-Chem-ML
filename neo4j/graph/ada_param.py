import pandas as pd
from py2neo import Graph, Node, Relationship
import cypher  # Run cypher commands
import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes


def ada_param(csv, algor="ada"):
    """

    :param model_csv:
    :param algor:
    :return:
    """

    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    graph_df = gd.GraphDataframe()
    label_df = graph_df.label_dataframe(csv, algor)
    param_dct = label_df.to_dict('records')  # Dict of dataframe for ml parameters
    for i in range(len(param_dct)):
        tx = graph.begin()
        print('Creating model nodes number: ' + str(i))
        loop_param = param_dct[i]
        regressor = Node("regress", regressor=loop_param['regressor'])
        tx.create(regressor)
        learning_rate_label = Node("learning_rate_label", learn_rate_label=loop_param['learning_rateRun#'])
        tx.create(learning_rate_label)
        learning_rate = Node("learning_rate", learning_rates=loop_param['learning_rate'])
        tx.create(learning_rate)
        n_estimators_label = Node("nestimators_label", nestimators_label=loop_param['n_estimatorsRun#'])
        tx.create(n_estimators_label)
        n_estimators = Node("n_estimators", estimators=loop_param['n_estimators'])
        tx.create(n_estimators)
        bf = Relationship(regressor, "has", learning_rate_label)
        tx.merge(bf)
        bg = Relationship(learning_rate_label, "is", learning_rate)
        tx.merge(bg)
        bh = Relationship(regressor, "has", n_estimators_label)
        tx.merge(bh)
        bi = Relationship(n_estimators_label, "is", n_estimators)
        tx.merge(bi)
        tx.commit()
