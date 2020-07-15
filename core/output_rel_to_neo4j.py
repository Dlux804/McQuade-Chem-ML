"""
Objective: Gather data from _data.csv and _attributes.csv files and import relationships based on our ontology to Neo4j
"""


from py2neo import Graph

# TODO: Add documentation


# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def relationships(prep):
    """
    Create relationships in Neo4j
    """
    print("Creating relationships...")

    # Merge RandomSplit node
    g.evaluate(" MATCH (n:RandomSplit) WITH n.test_percent AS test, n.train_percent as train, n.random_seed as seed,"
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

    # Merge TrainSet node
    g.evaluate(" MATCH (n:TrainSet) WITH n.trainsize as trainsize, n.random_seed as seed,"
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

    # Merge ValSet node
    g.evaluate(" MATCH (n:ValSet) WITH n.valsize as valsize, n.random_seed as seed,"
               "COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 "
               "CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

    # Merge Dataset with molecules and RandomSplit
    g.evaluate("""
                UNWIND $molecules as mol
                match (smiles:Molecule {SMILES:mol}), (dataset:DataSet {data: $dataset}), 
                (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, random_seed: $random_seed})
                merge (dataset)-[:CONTAINS_MOLECULES]->(smiles)
                merge (split)-[:USES_SPLIT]->(dataset)
        """, parameters={"molecules": prep.canonical_smiles, 'dataset': prep.dataset_str, 'test_percent': prep.test_percent,
                         'train_percent': prep.train_percent, 'random_seed': prep.random_seed})

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
    g.evaluate("""
                    UNWIND $feat_name as feat
                    match (method:FeatureMethod {feature: feat}), (model:MLModel {name: $run_name}) 
                   merge (model)-[:USES_FEATURIZATION]->(method)""",
                   parameters={'feat_name': prep.feat_method_name, 'run_name': prep.run_name})

    # RandomSplit to TrainSet and TestSet
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent}),"
               "(testset:TestSet {testsize: $test_size, RMSE: $rmse}), "
               "(trainset:TrainSet {trainsize: $training_size})"
               "merge (trainset)<-[:SPLITS_INTO]-(split)-[:SPLITS_INTO]->(testset)",
               parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                           'test_size': prep.n_test, 'rmse': prep.rmse, 'training_size': prep.n_train})
    
    # Merge Dataset to TrainSet and TestSet
    g.evaluate("""
            MATCH (dataset:DataSet {data: $dataset}), (trainset:TrainSet {trainsize: $training_size, 
            random_seed: $random_seed}), (testset:TestSet {testsize: $test_size, RMSE: $rmse})
            MERGE (trainset)<-[:SPLITS_INTO]-(dataset)-[:SPLITS_INTO]->(testset)
        """, parameters={'dataset': prep.dataset_str, 'training_size': prep.n_train, 'random_seed': prep.random_seed,
                         'test_size': prep.n_test, 'rmse': prep.rmse})

    # RandomSplit to Validation
    g.evaluate("match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, "
               "random_seed: $random_seed}), "
               "(validate:ValidateSet {valsize: $val_size}) merge (split)-[:SPLITS_INTO]->(validate)",
               parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                           'val_size': prep.n_val, 'random_seed': prep.random_seed})

    # Merge TrainSet with its molecules
    g.evaluate("""
                       UNWIND $train_smiles as mol
                       match (smile:Molecule {SMILES:mol}), (trainset:TrainSet {trainsize: $training_size, 
                       random_seed: $random_seed})
                       merge (smile)-[:CONTAINS_MOLECULES]->(trainset)""",
               parameters={'train_smiles': prep.train_molecules, 'training_size': prep.n_train,
                           'random_seed': prep.random_seed})

    # Merge TestSet with its molecules
    g.evaluate("""
                UNWIND $parameters as row
                match (smiles:Molecule {SMILES:row.smiles}), (testset:TestSet {testsize: $test_size, RMSE: $rmse})
                merge (testset)-[:CONTAINS_MOLECULES {predicted_value: row.predicted, uncertainty:row.uncertainty}]->(smiles)
            """, parameters={'parameters': prep.test_mol_dict, 'test_size': prep.n_test, 'rmse': prep.rmse})

    # Connect ValidateSet with its molecules
    if prep.val_percent > 0:
        # Merge RandomSplit to Validation
        g.evaluate("""match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
                      random_seed: $random_seed}), (validate:ValSet {valsize: $val_size, random_seed: $random_seed}) 
                      merge (split)-[:SPLITS_INTO]->(validate)""",
                      parameters={'test_percent': prep.test_percent, 'train_percent': prep.train_percent,
                               'val_size': prep.n_val, 'random_seed': prep.random_seed})
        
        # MLModel to ValidateSet
        g.evaluate("""match (validate:ValSet {valsize: $val_size, random_seed: $random_seed}), 
                           (model:MLModel {name: $run_name}) merge (model)-[:VALIDATE]->(validate)""",
                   parameters={'val_size': prep.n_val, 'run_name': prep.run_name, 'random_seed': prep.random_seed})
        
        # Merge ValidateSet with its molecules
        g.evaluate("""
                                UNWIND $val_smiles as mol
                                match (smile:Molecule {SMILES:mol}), (validate:ValSet {valsize: $val_size, 
                                random_seed: $random_seed}) merge (smile)-[:CONTAINS_MOLECULES]->(validate)""",
                   parameters={'val_smiles': prep.val_molecules, 'val_size': prep.n_val,
                               'random_seed': prep.random_seed})
        # Merge Validate with DataSet
        g.evaluate("""match (validate:ValSet {valsize: $val_size, random_seed: $random_seed}), 
                                (dataset:DataSet {data: $dataset})
                              merge (validate)-[:SPLITS_INTO]->(dataset)
                """, parameters={'val_size': prep.n_val, 'dataset': prep.dataset_str, 'random_seed': prep.random_seed})
    else:
        pass

    #     # Merge "SPLITS_INTO" relationship between RandomSplit and TrainSet
    # g.evaluate("""MATCH (:RandomSplit {test_percent: $test_percent, train_percent: $train_percent,
    #               random_seed: $random_seed})-[r:SPLITS_INTO]->(:TrainSet {trainsize:%f, random_seed: $random_seed})
    #               WITH r.name AS name, COLLECT(r) AS rell, COUNT(*) AS count
    #               WHERE count > 1
    #               CALL apoc.refactor.mergeRelationships(rell) YIELD rel
    #               RETURN rel""" % prep.n_train, parameters={'test_percent': prep.test_percent, 'train_percent':
    #         prep.train_percent, 'random_seed': prep.random_seed})
