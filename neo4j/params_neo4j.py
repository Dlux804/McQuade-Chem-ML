import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
import model_kg_v4 as v4
import extract_parmas as ep


def graph_gdbparam(model_csv, param_csv, algo="gdb"):
    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    model_df = pd.read_csv(model_csv) # csv file with all ml results
    param_df = pd.read_csv(param_csv)
    algo_df = model_df[model_df.algorithm == algo] # dataframe with specific algorithms
    regress_lst = v4.ML_kg(algo_df) # List of all the regressor nodes
    param_dct = param_df.to_dict('records') # Dict of dataframe for ml parameters
    for i in range(len(param_dct)):
        loop_param = param_dct[i]
        tx = graph.begin()
        learning_rate = Node("learning_rate", learn_rates=loop_param['learning_rate'])
        tx.create(learning_rate)
        max_depth = Node("max_depth", max_d=loop_param['max_depth'])
        tx.create(max_depth)
        max_features = Node("max_features", max_feat=loop_param['max_features'])
        tx.create(max_features)
        min_samples_leaf = Node("min_samples_leaf", min_leaf=loop_param['min_samples_leaf'])
        tx.create(min_samples_leaf)
        min_samples_split = Node("min_samples_split", min_split=loop_param['min_samples_split'])
        tx.create(min_samples_split)
        n_estimators = Node("n_estimators", estimators=loop_param['n_estimators'])
        tx.create(n_estimators)
        for regress in regress_lst:
            aa = Relationship(learning_rate, "learning_rate", regress)
            tx.merge(aa)
            ab = Relationship(max_depth, "max_depth", regress)
            tx.merge(ab)
            ac = Relationship(max_features, "max_features", regress)
            tx.merge(ac)
            ad = Relationship(min_samples_leaf, "min_samples_leaf", regress)
            tx.merge(ad)
            ae = Relationship(min_samples_split, "min_samples_split", regress)
            tx.merge(ae)
            af = Relationship(n_estimators, "n_estimators", regress)
            tx.merge(af)
            tx.commit()


graph_gdbparam('ml_results2.csv', 'gdb_params.csv')






