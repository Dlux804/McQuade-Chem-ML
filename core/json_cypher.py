"""
Objective: Create Cypher lines that import data from input files to Neo4j graphs
"""

from py2neo import Graph, Node

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def non_loop_graph(data_list, feature_length, test_time, train_size, test_size):
    """

    :param data_list:
    :param feature_length:
    :param test_time:
    :param train_size:
    :param test_size:
    :return:
    """
    # TODO: Add test_time, r2, rmse, mse
    query_line = """ 
        UNWIND $parameters as rows

        merge (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time,
                                test_time:%f})
        merge (featurelist:FeatureList {name:"FeatureList", num:%d})
        merge (trainset:TrainSet {trainsize:%f, name:"TrainSet"})
        merge (testset:TestSet {name:"TestSet", testsize:%f})
        merge (dataset:DataSet {name:"Dataset", source:"Moleculenet", data:rows.dataset, measurement:rows.target_name})
        merge (randomsplit:RandomSplit {name:"RandomSplit", test_percent:rows.test_percent,
                       train_percent:rows.train_percent, random_seed:rows.random_seed,
                       val_percent:rows.val_percent})
        merge (model)-[:USES_DATASET]->(dataset)
        merge (model)-[:USES_FEATURES]->(featurelist)
        merge (model)-[:USES_SPLIT]->(randomsplit)
        merge (model)-[:TRAINS]->(trainset)
        merge (model)-[:TRAINS]->(testset)
        merge (trainset)<-[:SPLITS_INTO]-(randomsplit)-[:SPLITS_INTO]->(testset)
        
    """ % (feature_length, test_time, train_size, test_size)

    g.evaluate(query_line, parameters={'parameters': data_list})


def if_nodes(data_list, algorithm, tuned, val_size):
    """

    :param data_list:
    :param algorithm:
    :param tuned:
    :param val_size:
    :return:
    """
    # Algorithm
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
        merge (model)-[:USES_ALGORITHM {rows.params}]->(algor)
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
    print(val_size)

    # Validation
    if val_size > 0.0:
        val_line = """
        UNWIND $parameters as rows
        match (model:MLModel {name:rows.run_name, feat_time:rows.feat_time, date:rows.date, train_time:rows.tune_time}),
        (randomsplit:RandomSplit {name:"RandomSplit", test_percent:rows.test_percent,
                       train_percent:rows.train_percent, random_seed:rows.random_seed})
        merge (validate:ValidateSet {name:"ValidateSet", valsize:%f})
        merge (model)-[:VALIDATE]->(validate)
        merge (randomsplit)-[:SPLITS_INTO]->(validate)
        
        """ % val_size
    else:
        pass


def feature_method_node(feat_name, feature_length):
    """"""
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






