import pandas as pd
from py2neo import Graph, Node, Relationship
import cypher  # Run cypher commands
import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes


def rf_param(csv, algor="rf"):
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
        loop_param = param_dct[i]
        regressor = Node("regress", regressor=loop_param['regressor'])
        tx.create(regressor)
        bootstrap_label = Node("bootstrap_label", bootstrap_labels=loop_param['bootstrapRun#'])
        tx.create(bootstrap_label)
        bootstrap = Node("bootstrap", bootstraps=loop_param['bootstrap'])
        tx.create(bootstrap)
        maxdepth_label = Node("maxdepth_label", maxdepth_labels=loop_param['max_depthRun#'])
        tx.create(maxdepth_label)
        maxdepth = Node("maxdepth", maxdepths=loop_param['max_depth'])
        tx.create(maxdepth)
        max_features_label = Node("max_features_label", max_features_labels=loop_param['max_featuresRun#'])
        tx.create(max_features_label)
        max_features = Node("max_features", max_feature=loop_param['max_features'])
        tx.create(max_features)
        n_estimators_label = Node("nestimators_label", nestimators_label=loop_param['n_estimatorsRun#'])
        tx.create(n_estimators_label)
        n_estimators = Node("n_estimators", estimators=loop_param['n_estimators'])
        tx.create(n_estimators)
        bf = Relationship(regressor, "has", bootstrap_label)
        tx.merge(bf)
        bg = Relationship(bootstrap_label, "is", bootstrap)
        tx.merge(bg)
        bh = Relationship(regressor, "has", maxdepth_label)
        tx.merge(bh)
        bi = Relationship(maxdepth_label, "is", maxdepth)
        tx.merge(bi)
        bj = Relationship(regressor, "has", max_features_label)
        tx.merge(bj)
        bk = Relationship(max_features_label, "is", max_features)
        tx.merge(bk)
        bl = Relationship(regressor, "has", n_estimators_label)
        tx.merge(bl)
        bm = Relationship(n_estimators_label, "is", n_estimators)
        tx.merge(bm)
        tx.commit()


# graph_rfparam('ml_results3.csv')
# gdb_param.graph_gdbparam('ml_results3.csv')
# cypher.run_cypher_command(pd.read_csv('ml_results3.csv'), "target")