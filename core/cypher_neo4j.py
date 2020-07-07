"""
Objective: Create Cypher lines that import data from input files to Neo4j graphs
"""


def sklearn_algorithm():
    query_line = """ 
        UNWIND $parameters as rows
        merge (algor:Algorithm {name: rows.algorithm, source: "sklearn", tuned: rows.tuned})
        merge (tuning_alg:TuningAlg {tuneTime= rows.tune_time, name: "TuningAlg", 
                                    algorithm="BayesianOptimizer", num_cv=self.cv_folds, tuneTime=self.tune_time, 
                                    delta=self.cp_delta, n_best=self.cp_n_best, steps=self.opt_iter})
        
    """
    return query_line


def nn_algorithm():
    query_line = """ 
            UNWIND $parameters as rows
            merge (algor:Algorithm {name=rows.algorithm, source="", tuned=rows.tuned})
        """
    return query_line