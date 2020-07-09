"""
Objective: Gather data from _data.csv and _attributes.csv files and import relationships based on our ontology to Neo4j
"""


from py2neo import Graph
from core.prep_from_outputs import split_molecules

# TODO: Add documentation


# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def relationships(prep):
    """
    Create relationships in Neo4j
    """
    print("Creating relationships...")

    # Merge RandomSplit node
    g.evaluate(" MATCH (n:RandomSplit) WITH n.test_percent AS test, n.train_percent as train, "
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

    # MLModel to Algorithm
    param_list = prep.params_list
    if prep.tuned:  # If tuned
        for param in param_list:
            try:
                g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
                           "merge (model)-[r:USES_ALGORITHM]->(algor) Set r.%s = $params" % param,
                           parameters={'algorithm': prep.algorithm, 'run_name': prep.run_name,
                                   'params': str(prep.df_from_attributes[param])})
            except TypeError:
                g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
                           "merge (model)-[r:USES_ALGORITHM]->(algor) Set r.%s = $params" % param,
                           parameters={'algorithm': prep.algorithm, 'run_name': prep.run_name,
                                       'params': float(prep.df_from_attributes[param])})
    else:  # If not tuned
        g.evaluate("match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})"
                   "merge (model)-[r:USES_ALGORITHM]->(algor)",
                   parameters={'algorithm': prep.algorithm, 'run_name': prep.run_name})

    # Algorithm to TuningAlg
    if prep.tuned:  # If tuned
        g.evaluate("match (tuning_alg:TuningAlg {tuneTime: $tune_time}), (algor:Algorithm {name: $algorithm})"
                   "merge (algor)-[:USES_TUNING]->(tuning_alg)",
                   parameters={'tune_time': prep.tune_time, 'algorithm': prep.algorithm})
    else:
        pass  # not tuned
    # MLModel to DataSet
    g.evaluate("match (dataset:DataSet {data: $dataset}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_DATASET]->(dataset)",
               parameters={'dataset': prep.dataset_str, 'run_name': prep.run_name})

    # MLModel to Featurelist
    g.evaluate("match (featurelist:FeatureList {num: $feat_length}), (model:MLModel {name: $run_name})"
               "merge (model)-[:USES_FEATURES]->(featurelist)",
               parameters={'feat_length': prep.num_feature_list, 'run_name': prep.run_name})

    # MLModel to RandomSplit
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, "
               "random_seed: $random_seed}), "
               "(model:MLModel {name: $run_name})"
               "merge (model)-[:USES_SPLIT]->(split)",
               parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                           'run_name': prep.run_name, 'random_seed': prep.random_seed})

    # MLModel to TrainingSet
    g.evaluate("match (trainset:TrainSet {trainsize: $training_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:TRAINS]->(trainset)",
               parameters={'training_size': prep.n_train, 'run_name': prep.run_name})

    # MLModel to TestSet
    g.evaluate("match (testset:TestSet {testsize: $test_size, RMSE: $rmse}), (model:MLModel {name: $run_name})"
               "merge (model)-[:PREDICTS]->(testset)",
               parameters={'test_size': prep.n_test, 'rmse': prep.rmse, 'run_name': prep.run_name})

    # MLModel to ValidateSet
    g.evaluate("match (validate:ValidateSet {valsize: $val_size}), (model:MLModel {name: $run_name})"
               "merge (model)-[:VALIDATE]->(validate)",
               parameters={'val_size': prep.n_val, 'run_name': prep.run_name})

    # MLModel to feature method, FeatureList to feature method
    for feat in prep.feat_method_name:
        g.evaluate("match (method:FeatureMethod {feature: $feat}), (model:MLModel {name: $run_name}) "
                   "merge (model)-[:USES_FEATURIZATION]->(method)",
                   parameters={'feat': feat, 'run_name': prep.run_name})
        g.evaluate("match (method:FeatureMethod {feature: $feat}), (featurelist:FeatureList {num: $feat_length}) "
                   "merge (method)-[:CONTRIBUTES_TO]->(featurelist)",
                   parameters={'feat': feat, 'feat_length': prep.num_feature_list})

    # RandomSplit to TrainSet and TestSet
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}),"
               "(testset:TestSet {testsize: $test_size, RMSE: $rmse}), "
               "(trainset:TrainSet {trainsize: $training_size})"
               "merge (trainset)<-[:SPLITS_INTO]-(split)-[:SPLITS_INTO]->(testset)",
               parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                           'test_size': prep.n_test, 'rmse': prep.rmse, 'training_size': prep.n_train})
    # RandomSplit to Validation
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, "
               "random_seed: $random_seed}), "
               "(validate:ValidateSet {valsize: $val_size}) merge (split)-[:SPLITS_INTO]->(validate)",
               parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                           'val_size': prep.n_val, 'random_seed': prep.random_seed})

    # Connect TrainSet with its molecules
    train_molecules = split_molecules(prep.df_from_data, 'train')
    for train_smiles in list(train_molecules):
        g.evaluate("match (smile:SMILES {SMILES:$mol}), (trainset:TrainSet {trainsize: $training_size})"
                   "merge (smile)-[:CONTAINS_MOLECULES]->(trainset)",
                   parameters={'mol': train_smiles, 'training_size': prep.n_train})

    # Connect TestSet with its molecules
    test_molecules = split_molecules(prep.df_from_data, 'test')
    for test_smiles, predict in zip(test_molecules, prep.predicted):
        g.evaluate("match (smiles:SMILES {SMILES:$mol}), (testset:TestSet {testsize: $test_size, RMSE: $rmse})"
                   "merge (testset)-[:CONTAINS_MOLECULES {predicted_value: $predicted}]->(smiles)",
                   parameters={'mol': test_smiles, 'test_size': prep.n_test, 'rmse': prep.rmse, 'predicted': predict})

    # Connect ValidateSet with its molecules

    if prep.val_percent > 0:
        val_molecules = split_molecules(prep.df_from_data, 'val')
        for val_smiles in list(val_molecules):
            g.evaluate("match (smile:SMILES {SMILES:$mol}), (validate:ValidateSet {valsize: $val_size})"
                       "merge (smile)-[:CONTAINS_MOLECULES]->(validate)",
                       parameters={'mol': val_smiles, 'val_size': prep.n_val})
    else:
        pass

    # Merge "SPLITS_INTO" relationship between RandomSplit and TrainSet
    g.evaluate("""MATCH (:RandomSplit)-[r:SPLITS_INTO]-(:TrainSet {trainsize:%d})
                WITH r.name AS name, COLLECT(r) AS rell, COUNT(*) AS count
                WHERE count > 1
                CALL apoc.refactor.mergeRelationships(rell) YIELD rel
                RETURN rel""" % prep.n_train)

