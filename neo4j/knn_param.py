import pandas as pd
from py2neo import Graph, Node, Relationship
import cypher  # Run cypher commands
import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes


def graph_knnparam(model_csv, algor="knn"):
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
        print('Creating Relationships Number ' + str(i))
        tx = graph.begin()
        loop_param = param_dct[i]
        regressor = Node("regress", regressor=loop_param['regressor'])
        tx.create(regressor)
        leaf_size_label = Node("leaf_size_label", leaf_size_labels=loop_param['leaf_sizeRun#'])
        tx.create(leaf_size_label)
        leaf_size = Node("leaf_size", leaf_sizes=loop_param['leaf_size'])
        tx.create(leaf_size)
        n_neighbors_label = Node("n_neighbors_label", n_neighbors_labels=loop_param['n_neighborsRun#'])
        tx.create(n_neighbors_label)
        n_neighbors = Node("n_neighbors", n_neighbor=loop_param['n_neighbors'])
        tx.create(n_neighbors)
        p_label = Node("p_label", p_labels=loop_param['pRun#'])
        tx.create(p_label)
        p = Node("p", ps=loop_param['p'])
        tx.create(p)
        weights_label = Node("weights_label", weights_labels=loop_param['weightsRun#'])
        tx.create(weights_label)
        weights = Node("weights", weight=loop_param['weights'])
        tx.create(weights)
        bf = Relationship(regressor, "has", leaf_size_label)
        tx.merge(bf)
        bg = Relationship(leaf_size_label, "is", leaf_size)
        tx.merge(bg)
        bh = Relationship(regressor, "has", n_neighbors_label)
        tx.merge(bh)
        bi = Relationship(n_neighbors_label, "is", n_neighbors)
        tx.merge(bi)
        bj = Relationship(regressor, "has", p_label)
        tx.merge(bj)
        bk = Relationship(p_label, "is", p)
        tx.merge(bk)
        bl = Relationship(regressor, "has", weights_label)
        tx.merge(bl)
        bm = Relationship(weights_label, "is", weights)
        tx.merge(bm)
        tx.commit()
