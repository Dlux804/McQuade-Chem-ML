"""
Objective: The goal of this script is to create relationships in Neo4j directly from the pipeline using class instances
"""
import time

from py2neo import Graph

from core.neo4j.nodes_to_neo4j import prep


# TODO REDO DOCSTRINGS


def relationships(self):
    """
    Objective: Create relationships in Neo4j based on our ontology directly from the pipeline.
    Intent: I want this script to only create relationships for the nodes in Neo4j. There are also some node merging
            lines sprinkled in since you can only merge on one property using py2neo. The reason why everything is
            broken into chunks is for readability and easier trouble shooting. I don't want to read a giant merge
            statements with 30+ variables, trying to figure out which Cypher correlates to what variable in Python
    """
    print("Creating relationships...")
    t1 = time.perf_counter()
    r2, mse, rmse, canonical_smiles, df_smiles, test_mol_dict = prep(self)
    g = Graph(self.neo4j_params["port"], username=self.neo4j_params["username"],
              password=self.neo4j_params["password"])  # Define graph for function

    # # Merge RandomSplit node
    # g.evaluate(""" MATCH (n:RandomSplit) WITH n.test_percent AS test, n.train_percent as train, n.random_seed as seed,
    #            COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
    #            CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")
    #
    # # Merge TrainSet node
    # g.evaluate(""" MATCH (n:TrainSet) WITH n.trainsize as trainsize, n.random_seed as seed,
    #            COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
    #            CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")
    #
    # # Merge ValSet node
    # g.evaluate(""" MATCH (n:ValSet) WITH n.valsize as valsize, n.random_seed as seed,
    #            COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
    #            CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node""")
    #
    # # Merge TestSet node
    # g.evaluate("""
    #         MATCH (n:TestSet) WITH n.testsize as testsize, n.random_seed as seed,
    #         COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1
    #         CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node
    #         """)

    # Merge Dataset with molecules and RandomSplit
    g.evaluate("""
            UNWIND $molecules as mol
            match (smiles:Molecule {SMILES:mol}), (dataset:DataSet {data: $dataset}), 
            (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, random_seed: $random_seed})
            merge (dataset)-[:CONTAINS_MOLECULES]->(smiles)
            merge (split)-[:USES_SPLIT]->(dataset)
    """, parameters={"molecules": canonical_smiles, 'dataset': self.dataset, 'test_percent': self.test_percent,
                     'train_percent': self.train_percent, 'random_seed': self.random_seed})
    # Merge FeatureList to FeatureMethod
    g.evaluate("""
        UNWIND $method_name as name
        match (method:FeatureMethod {feature:name}), (featurelist:FeatureList {feat_ID: $feat_ID})
        merge (featurelist)<-[:CONTRIBUTES_TO]-(method)
        """, parameters={'feat_ID': self.feat_meth, 'method_name': self.feat_method_name})

    # Merge MLModel to Algorithm
    if self.tuned:  # If tuned
        param_dict = dict(self.params)
        for key in param_dict:
            try:
                value = param_dict[key].values[0]
            except AttributeError:
                value = param_dict[key]
            g.evaluate("""
                        MATCH (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})
                        MERGE (model)-[r:USES_ALGORITHM]->(algor) 
                            SET r.%s = "%s" 
                        """ % (key, value),
                       parameters={'algorithm': self.algorithm, 'run_name': self.run_name, 'key': key}
                       )
            # except AttributeError:
            #     print(f"Failed to merge relationship with {key}, with a value of {param_dict[key]}")
    else:  # If not tuned
        g.evaluate("""match (algor:Algorithm {name: $algorithm}), (model:MLModel {name: $run_name})
                   merge (model)-[r:USES_ALGORITHM]->(algor)""",
                   parameters={'algorithm': self.algorithm, 'run_name': self.run_name})

    # Algorithm and MLModel to TuningAlg
    if self.tuned:  # If tuned
        g.evaluate("""match (tuning_alg:TuningAlg), (algor:Algorithm {name: $algorithm}), 
                    (model:MLModel {name: $run_name})
                   merge (algor)-[:USES_TUNING]->(tuning_alg)<-[r:TUNED_WITH]-(model)
                   Set r.num_cv=$cv_folds, r.tuneTime= $tune_time, r.delta = $cp_delta, r.n_best = $cp_n_best,
                                r.steps=$opt_iter
                   """,
                   parameters={'tune_time': self.tune_time, 'algorithm': self.algorithm, 'run_name': self.run_name,
                               'cv_folds': self.cv_folds, 'tunTime': self.tune_time, 'cp_delta': self.cp_delta,
                               'cp_n_best': self.cp_n_best, 'opt_iter': self.opt_iter})
    else:  # If not tuned
        g.evaluate("""match (tuning_alg:NotTuned), (algor:Algorithm {name: $algorithm})
                   merge (algor)-[:NOT_TUNED]->(tuning_alg)""",
                   parameters={'algorithm': self.algorithm})
    # MLModel to DataSet
    g.evaluate("""match (dataset:DataSet {data: $dataset}), (model:MLModel {name: $run_name})
               merge (model)-[:USES_DATASET]->(dataset)""",
               parameters={'dataset': self.dataset, 'run_name': self.run_name})

    # MLModel to Featurelist
    g.evaluate("""match (featurelist:FeatureList {num: $feat_length}), (model:MLModel {name: $run_name})
               merge (model)-[:USES_FEATURES]->(featurelist)""",
               parameters={'feat_length': self.feature_length, 'run_name': self.run_name})

    # MLModel to RandomSplit
    g.evaluate("""match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
               random_seed: $random_seed}), (model:MLModel {name: $run_name}) merge (model)-[:USES_SPLIT]->(split)""",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'random_seed': self.random_seed, 'run_name': self.run_name})

    # MLModel to TrainingSet
    g.evaluate("""match (trainset:TrainSet {trainsize: $training_size, random_seed: $random_seed}), 
               (model:MLModel {name: $run_name}) merge (model)-[:TRAINS]->(trainset)""",
               parameters={'training_size': self.n_train, 'run_name': self.run_name, 'random_seed': self.random_seed})

    # MLModel to TestSet
    g.evaluate("""
            match (testset:TestSet {testsize: $test_size, random_seed: $random_seed}), (model:MLModel {name: $run_name})
            merge (model)-[r:PREDICTS]->(testset)
            Set r.rmse = $rmse, r.mse = $mse, r.r2 = $r2
               """,
               parameters={'test_size': self.n_test, 'rmse': rmse, 'mse': mse, 'r2': r2, 'run_name': self.run_name,
                           'random_seed': self.random_seed})

    # MLModel to feature method, FeatureList to feature method
    g.evaluate("""
                        UNWIND $feat_name as feat
                        match (method:FeatureMethod {feature: feat}), (model:MLModel {name: $run_name}) 
                       merge (model)-[:USES_FEATURIZATION]->(method)""",
               parameters={'feat_name': self.feat_method_name, 'run_name': self.run_name})

    # Merge RandomSplit to TrainSet and TestSet
    g.evaluate("""match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
               random_seed: $random_seed}), (testset:TestSet {testsize: $test_size, random_seed: $random_seed}), 
               (trainset:TrainSet {trainsize: $training_size, random_seed: $random_seed})
               merge (trainset)<-[:MAKES_SPLIT]-(split)-[:MAKES_SPLIT]->(testset)""",
               parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                           'test_size': self.n_test, 'rmse': rmse, 'training_size': self.n_train,
                           'random_seed': self.random_seed})

    # Merge Dataset to TrainSet and TestSet
    g.evaluate("""
        MATCH (dataset:DataSet {data: $dataset}), (trainset:TrainSet {trainsize: $training_size, 
        random_seed: $random_seed}), (testset:TestSet {testsize: $test_size, random_seed: $random_seed})
        MERGE (trainset)<-[:SPLITS_INTO]-(dataset)-[:SPLITS_INTO]->(testset)
    """, parameters={'dataset':self.dataset, 'training_size': self.n_train, 'random_seed': self.random_seed,
                     'test_size': self.n_test, 'rmse': rmse})

    # Merge TrainSet with its molecules
    g.evaluate("""
                   UNWIND $train_smiles as mol
                   match (smiles:Molecule {SMILES:mol}), (trainset:TrainSet {trainsize: $training_size, 
                   random_seed: $random_seed})
                   merge (smiles)<-[:CONTAINS_MOLECULES]-(trainset)""",
                   parameters={'train_smiles': list(self.train_molecules), 'training_size': self.n_train,
                              'random_seed': self.random_seed})

    # Merge TestSet with its molecules
    g.evaluate("""
        UNWIND $parameters as row
        match (testset:TestSet {testsize: $test_size, random_seed: $random_seed}) 
        merge (smiles:Molecule {SMILES: row.smiles}) 
        merge (testset)-[:CONTAINS_MOLECULES {predicted_value: row.predicted, uncertainty:row.uncertainty}]->(smiles)
        """, parameters={'parameters': test_mol_dict, 'test_size': self.n_test, 'random_seed': self.random_seed})

    if self.val_percent > 0:
        # Merge RandomSplit to Validation
        g.evaluate("""match (split:RandomSplit {test_percent: $test_percent, train_percent: $train_percent, 
                   random_seed: $random_seed}), (validate:ValSet {valsize: $val_size, random_seed: $random_seed}) 
                   merge (split)-[:MAKES_SPLIT]->(validate)""",
                   parameters={'test_percent': self.test_percent, 'train_percent': self.train_percent,
                               'val_size': self.n_val, 'random_seed': self.random_seed})

        # MLModel to ValidateSet
        g.evaluate("""match (validate:ValSet {valsize: $val_size, random_seed: $random_seed}), 
                   (model:MLModel {name: $run_name}) merge (model)-[:VALIDATE]->(validate)""",
                   parameters={'val_size': self.n_val, 'run_name': self.run_name, 'random_seed': self.random_seed})

        # Merge ValidateSet with its molecules
        g.evaluate("""
                        UNWIND $val_smiles as mol
                        match (smiles:Molecule {SMILES: mol}), (validate:ValSet {valsize: $val_size, 
                        random_seed: $random_seed}) merge (validate)-[:CONTAINS_MOLECULES]->(smiles)""",
                   parameters={'val_smiles': list(self.val_molecules), 'val_size': self.n_val,
                               'random_seed': self.random_seed})

        # Merge Validate with DataSet
        g.evaluate("""match (validate:ValSet {valsize: $val_size, random_seed: $random_seed}), 
                        (dataset:DataSet {data: $dataset})
                      merge (dataset)-[:SPLITS_INTO]->(validate)
        """, parameters={'val_size': self.n_val, 'dataset':self.dataset, 'random_seed': self.random_seed})
    else:
        pass
    t2 = time.perf_counter()
    print(f"Time it takes to create the rest of the relationships: {t2-t1}")