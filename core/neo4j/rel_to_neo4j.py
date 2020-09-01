"""
Objective: The goal of this script is to create relationships in Neo4j directly from the pipeline using class instances
"""

from py2neo import Graph
import time
from core.neo4j.make_query import Query
import ast
from core.neo4j.nodes_to_neo4j import prep


# TODO REDO DOCSTRINGS


def relationships(self, from_output=False):
    """
    Objective: Create relationships in Neo4j based on our ontology directly from the pipeline.
    Intent: I want this script to only create relationships for the nodes in Neo4j. There are also some node merging
            lines sprinkled in since you can only MERGE on one property using py2neo. The reason why everything is
            broken into chunks is for readability and easier trouble shooting. I don't want to read a giant MERGE
            statements with 30+ variables, trying to figure out which Cypher correlates to what variable in Python
    """
    print("Creating relationships...")
    t1 = time.perf_counter()
    self.tuned = str(self.tuned).capitalize()

    df_smiles, test_mol_dict, data_size, cv_results = prep(self)
    g = Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Define graph for function
    query = Query(graph=g)
    query.__check_for_constraints__()
    # MERGE RandomSplit node
    g.evaluate(""" MATCH (n:RandomSplit) WITH n.test_percent AS test, n.train_percent as train, n.random_seed as seed,
                   COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
                   CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")

    # MERGE Dataset with RandomSplit
    g.evaluate("""
            MATCH (dataset:DataSet {data: $dataset}), 
            (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, random_seed: $random_seed})
            MERGE (split)-[:SPLITS_DATASET]->(dataset)
    """, parameters={'dataset': self.dataset, 'test_percent': self.test_percent,
                     'train_percent': self.train_percent, 'random_seed': self.random_seed})
    # MERGE FeatureList to FeatureMethod
    g.evaluate("""
        UNWIND $method_name as name
        MATCH (method:FeatureMethod {feature:name}), (featurelist:FeatureList {feat_ID: $feat_ID})
        MERGE (featurelist)<-[:CONTRIBUTES_TO]-(method)
        """, parameters={'feat_ID': self.feat_meth, 'method_name': self.feat_method_name})

    # MERGE MLModel to Algorithm
    if ast.literal_eval(self.tuned):  # If tuned
        if not from_output:
            self.params = dict(self.params)
        try:
            g.evaluate("""
                MATCH (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})
                MERGE (model)-[r:USES_ALGORITHM]->(algor) Set r = $param, r.tuned = $tuned """,
                       parameters={'algorithm': self.algorithm, 'run_name': self.run_name, 'param': self.params,
                                   'tuned': self.tuned})
        except TypeError:  # Adaboost's parameter causing problem
            self.params['base_estimator'] = str(self.params['base_estimator'])  # Turn based_estimator to string
            g.evaluate("""
                    MATCH (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})
                    MERGE (model)-[r:USES_ALGORITHM]->(algor) Set r = $param, r.tuned = $tuned """,
                       parameters={'algorithm': self.algorithm, 'run_name': self.run_name, 'param': self.params,
                                   'tuned': self.tuned})
    else:  # If not tuned
        g.evaluate("""MATCH (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})
                   MERGE (model)-[r:USES_ALGORITHM]->(algor) Set r.tuned = $tuned""",
                   parameters={'algorithm': self.algorithm, 'run_name': self.run_name, 'tuned': self.tuned})

    # Algorithm and MLModel to TuningAlg
    if ast.literal_eval(self.tuned):  # If tuned
        g.evaluate("""
                    MATCH (tuning_alg:TuningAlg {algorithm:$tuner}), (algor:Algorithm {name: $algorithm}), 
                    (model:MLModel {name: $run_name})
                   MERGE (algor)-[:USES_TUNING]->(tuning_alg)
                   MERGE (model)-[r:TUNED_WITH]->(tuning_alg)
                   Set r.num_cv=$cv_folds, r.tuneTime= $tune_time, r.delta = $cp_delta, r.n_best = $cp_n_best,
                                r.steps=$opt_iter
                   Set r += $cv_results
                   """,
                   parameters={'tune_time': self.tune_time, 'algorithm': self.algorithm, 'run_name': self.run_name,
                               'cv_folds': self.cv_folds, 'tunTime': self.tune_time, 'cp_delta': self.cp_delta,
                               'cp_n_best': self.cp_n_best, 'opt_iter': self.opt_iter, 'tuned': self.tuned,
                               'tuner': self.tune_algorithm_name, 'cv_results': cv_results})

    # MLModel to DataSet
    g.evaluate("""MATCH (dataset:DataSet {data: $dataset}), (model:MLModel {name: $run_name})
               MERGE (model)-[:USES_DATASET]->(dataset)""",
               parameters={'dataset': self.dataset, 'run_name': self.run_name})

    # MLModel to Featurelist
    g.evaluate("""MATCH (featurelist:FeatureList {feat_ID: $feat_meth}), (model:MLModel {name: $run_name})
               MERGE (model)-[:USES_FEATURES]->(featurelist)""",
               parameters={'feat_meth': self.feat_meth, 'run_name': self.run_name})

    # MLModel to RandomSplit
    g.evaluate("""MATCH (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
               random_seed: $random_seed}), (model:MLModel {name: $run_name}) MERGE (model)-[:USES_SPLIT]->(split)""",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'random_seed': self.random_seed, 'run_name': self.run_name})

    # MLModel to TrainingSet
    g.evaluate("""MATCH (trainset:TrainSet {run_name: $run_name}), (model:MLModel {name: $run_name}) 
                  MERGE (model)-[:TRAINS]->(trainset)""",
               parameters={'run_name': self.run_name})

    # MLModel to TestSet
    g.evaluate("""
            MATCH (testset:TestSet {run_name: $run_name}), (model:MLModel {name: $run_name})
            MERGE (model)-[r:PREDICTS]->(testset)
            Set r.rmse_avg = $rmse, r.mse_avg = $mse, r.r2_avg = $r2
               """,
               parameters={'rmse': self.predictions_stats['rmse_avg'],
                           'mse': self.predictions_stats['mse_avg'], 'r2': self.predictions_stats['r2_avg'],
                           'run_name': self.run_name})

    # MLModel to feature method
    g.evaluate("""
                        UNWIND $feat_name as feat
                        MATCH (method:FeatureMethod {feature: feat}), (model:MLModel {name: $run_name}) 
                       MERGE (model)-[:USES_FEATURIZATION]->(method)""",
               parameters={'feat_name': self.feat_method_name, 'run_name': self.run_name})

    # MERGE RandomSplit to TrainSet and TestSet
    g.evaluate("""MATCH (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
               random_seed: $random_seed}), (testset:TestSet {run_name: $run_name}), 
               (trainset:TrainSet {run_name: $run_name})
               MERGE (trainset)<-[:MAKES_TRAIN_SPLIT]-(split)-[:MAKES_TEST_SPLIT]->(testset)""",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent, 
                           'run_name': self.run_name, 'random_seed': self.random_seed})

    # MERGE Dataset to TrainSet and TestSet
    g.evaluate("""
        MATCH (dataset:DataSet {data: $dataset}), (trainset:TrainSet {run_name: $run_name}), 
              (testset:TestSet {run_name: $run_name})
        MERGE (trainset)<-[:SPLITS_INTO_TRAIN]-(dataset)-[:SPLITS_INTO_TEST]->(testset)
    """, parameters={'dataset': self.dataset, 'run_name': self.run_name})

    # MERGE TrainSet with its molecules
    g.evaluate("""
        UNWIND $train_smiles as mol
        MATCH (trainset:TrainSet {run_name: $run_name}), (smiles:Molecule {SMILES:mol})
        MERGE (smiles)<-[:CONTAINS_TRAINED_MOLECULES]-(trainset)""",
               parameters={'train_smiles': list(self.train_molecules), 'run_name': self.run_name})

    # MERGE TestSet with its molecules
    g.evaluate("""
    UNWIND $parameters as row
    MATCH (testset:TestSet {run_name: $run_name}), (smiles:Molecule {SMILES: row.smiles}) 
    MERGE (testset)-[:PREDICTS_MOL_PROP {average_prediction: row.average_prediction, uncertainty:row.uncertainty, 
                                            average_error: row.average_error}]->(smiles)
        """, parameters={'parameters': test_mol_dict, 'run_name': self.run_name})

    if self.val_percent > 0:
        # MERGE RandomSplit to Validation
        g.evaluate("""MATCH (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
                   random_seed: $random_seed}), (valset:ValSet {run_name: $run_name}) 
                   MERGE (split)-[:MAKES_VAL_SPLIT]->(valset)""",
                   parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                               'run_name': self.run_name, 'random_seed': self.random_seed})

        # MLModel to ValidateSet
        g.evaluate("""MATCH (valset:ValSet {run_name: $run_name}), 
                   (model:MLModel {name: $run_name}) MERGE (model)-[:VALIDATE]->(valset)""",
                   parameters={'run_name': self.run_name})

        # MERGE ValidateSet with its molecules
        g.evaluate("""
                        UNWIND $val_smiles as mol
                        MATCH (smiles:Molecule {SMILES: mol}), (valset:ValSet {run_name: $run_name}) 
                        MERGE (valset)-[:CONTAINS_VALIDATED_MOLECULES]->(smiles)""",
                   parameters={'val_smiles': list(self.val_molecules), 'run_name': self.run_name})

        # MERGE Validate with DataSet
        g.evaluate("""MATCH (valset:ValSet {run_name: $run_name}), (dataset:DataSet {data: $dataset})
                      MERGE (dataset)-[:SPLITS_INTO_VAL]->(valset)
        """, parameters={'run_name': self.run_name, 'dataset': self.dataset})
    else:
        pass
    t2 = time.perf_counter()
    print(f"Time it takes to create the rest of the relationships: {t2 - t1}sec")
