"""
Objective: Create Cypher lines that import data from _attributes.json files to Neo4j graphs
"""

from py2neo import Graph, Node

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def non_loop_graph(data_list, df_from_attributes):
    """"""
    feature_length = len(df_from_attributes['feature_list'])  # feature length
    test_time = df_from_attributes["predictions_stats.time_avg"]  # test time
    query_line = """ 
        UNWIND $parameters as rows

        merge (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time,
                                test_time:%f})
        merge (featurelist:FeatureList {name:"FeatureList", num:%d})
        merge (dataset:DataSet {name:"Dataset", source:"Moleculenet", data:rows.dataset, target_col:rows.target_name})
        merge (randomsplit:RandomSplit {name:"RandomSplit", test_percent:rows.test_percent,
                       train_percent:rows.train_percent, random_seed:rows.random_seed,
                       val_percent:rows.val_percent})
        merge (model)-[:USES_DATASET]->(dataset)
        merge (model)-[:USES_FEATURES]->(featurelist)
        merge (model)-[:USES_SPLIT]->(randomsplit)
        
    """ % (feature_length, test_time)

    g.evaluate(query_line, parameters={'parameters': data_list})


def if_nodes(data_list, df_from_attributes):
    """"""
    # Algorithm
    algorithm = str(df_from_attributes['algorithm'])  # algorithm
    tuned = str(df_from_attributes['tuned'])  # tuned or not
    if algorithm == "nn":
        algorithm_line = """
        UNWIND $parameters as rows
        match (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time})
        merge (algor:Algorithm {name: rows.algorithm, source: "keras", tuned: rows.tuned})
        merge (model)-[:USES_ALGORITHM]->(algor)
        """
    else:
        algorithm_line = """
        UNWIND $parameters as rows
        match (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time})
        merge (algor:Algorithm {name: rows.algorithm, source: "sklearn", tuned: rows.tuned})
        merge (model)-[:USES_ALGORITHM]->(algor)
                """
    g.evaluate(algorithm_line, parameters={'parameters': data_list})

    # Tuned
    if tuned:
        tune_line = """
        UNWIND $parameters as rows
        match (algor:Algorithm {name: rows.algorithm, tuned: rows.tuned})
        merge (tuningalg: TuningAlg {tuneTime: rows.tune_time, name: "TuningAlg", algorithm: "BayesianOptimizer",
        num_cv: rows.cv_folds, tuneTime: rows.tune_time, delta: rows.cp_delta, n_best: rows.cp_n_best, 
        steps: rows.opt_iter}) 
        merge (algor)-[:USES_TUNING]->(tuningalg)
        """
    else:
        tune_line = """
        UNWIND $parameters as rows
        match (algor:Algorithm {name: rows.algorithm, tuned: rows.tuned})
        merge (tuning_alg: NotTuned {name:"NotTuned"})
        merge (algor)-[:USES_TUNING]->(tuningalg)
        """
    g.evaluate(tune_line, parameters={'parameters': data_list})


def feature_method_node(df_from_attributes):
    """"""
    feat_name = list(df_from_attributes['feat_method_name'])  # feature method
    feature_length = len(df_from_attributes['feature_list'])  # feature length
    for feat in feat_name:
        feature_method = """
        UNWIND $parameters as rows
        match (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time}),
        (featurelist:FeatureList {name:"FeatureList", num:%d})
        merge (method:FeatureMethod {feature:%s, name:%s})
        merge (model)-[:USES_FEATURIZATION]->(method)
        merge (method)-[:CONTRIBUTES_TO]->(featurelist)
        """ % (feature_length, feat, feat)


# def train_test_molecules(train_molcules, test_molecules):
#     """"""






