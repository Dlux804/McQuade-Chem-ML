"""
Objective: Create Cypher lines that import data from input files to Neo4j graphs
"""


def non_loop_nodes(feature_length, test_time, train_size, test_size):
    # Add test_time, r2, rmse, mse
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
        merge ()
    """ % (feature_length, test_time, train_size, test_size)
    return query_line


def if_nodes(algorithm, tuned):
    """

    :param algorithm:
    :return:
    """

    if algorithm == "nn":
        algorithm_line = """
        UNWIND $parameters as rows
        merge (algor:Algorithm {name: rows.algorithm, source: "keras", tuned: rows.tuned})
        """
    else:
        algorithm_line = """
        UNWIND $parameters as rows
        merge (algor:Algorithm {name: rows.algorithm, source: "moleculenet.ai", tuned: rows.tuned})
                """
    print(tuned)
    if tuned == "True":
        tune_line = """ 
        UNWIND $parameters as rows
        merge (tuning_alg: TuningAlg {tuneTime: rows.tune_time, name: "TuningAlg", algorithm: "BayesianOptimizer", 
        num_cv: rows.cv_folds, tuneTime: rows.tune_time, delta: rows.cp_delta, n_best: rows.cp_n_best, steps: rows.opt_iter}) """
    else:
        tune_line = """ 
        merge (tuning_alg: NotTuned {name:"NotTuned"})
        """

    return algorithm_line, tune_line

def nn_algorithm():
    query_line = """ 
            UNWIND $parameters as rows
            merge (algor:Algorithm {name=rows.algorithm, source="", tuned=rows.tuned})
        """
    return query_line