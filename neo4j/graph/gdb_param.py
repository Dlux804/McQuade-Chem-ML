import pandas as pd
from py2neo import Graph, Node, Relationship
import cypher  # Run cypher commands
import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes


def gdb_param(csv, algor="gdb"):
    """

    :param csv:
    :param algor:
    :return:
    """
#
    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    graph_df = gd.GraphDataframe()
    label_df = graph_df.label_dataframe(csv, algor)
    param_dct = label_df.to_dict('records')  # Dict of dataframe for ml parameters
    for i in range(len(param_dct)):
        # graph params
        print('Creating param number: ' + str(i))
        tx = graph.begin()
        loop_param = param_dct[i]
        regressor = Node("regress", regressor=loop_param['regressor'])
        tx.create(regressor)
        learning_rate_label = Node("learningrate_label", learningrate_labels=loop_param["learning_rateRun#"])
        tx.create(learning_rate_label)
        learning_rate = Node("learning_rate", learn_rates=loop_param['learning_rate'])
        tx.create(learning_rate)
        max_depth_label = Node("maxdepth_label", maxdepth_label=loop_param["max_depthRun#"])
        tx.create(max_depth_label)
        max_depth = Node("max_depth", max_d=loop_param['max_depth'])
        tx.create(max_depth)
        max_features_label = Node("maxfeatures_label", maxfeatures_label=loop_param["max_featuresRun#"])
        tx.create(max_features_label)
        max_features = Node("max_features", max_feat=loop_param['max_features'])
        tx.create(max_features)
        min_samples_leaf_label = Node("minsamplesleaf_label", minleaf_label=loop_param["min_samples_leafRun#"])
        tx.create(min_samples_leaf_label)
        min_samples_leaf = Node("min_samples_leaf", min_leaf=loop_param['min_samples_leaf'])
        tx.create(min_samples_leaf)
        min_samples_split_label = Node("minsplit_label", minsplit_label=loop_param['min_samples_splitRun#'])
        tx.create(min_samples_split_label)
        min_samples_split = Node("min_samples_split", min_split=loop_param['min_samples_split'])
        tx.create(min_samples_split)
        n_estimators_label = Node("nestimators_label", nestimators_label=loop_param['n_estimatorsRun#'])
        tx.create(n_estimators_label)
        n_estimators = Node("n_estimators", estimators=loop_param['n_estimators'])
        tx.create(n_estimators)
        bf = Relationship(regressor, "has", learning_rate_label)
        tx.merge(bf)
        bg = Relationship(learning_rate_label, "is", learning_rate)
        tx.merge(bg)
        bh = Relationship(regressor, "has", max_depth_label)
        tx.merge(bh)
        bi = Relationship(max_depth_label, "is", max_depth)
        tx.merge(bi)
        bj = Relationship(regressor, "has", min_samples_split_label)
        tx.merge(bj)
        bk = Relationship(min_samples_split_label, "is", min_samples_split)
        tx.merge(bk)
        bl = Relationship(regressor, "has", min_samples_leaf_label)
        tx.merge(bl)
        bm = Relationship(min_samples_leaf_label, "is", min_samples_leaf)
        tx.merge(bm)
        tx.commit()