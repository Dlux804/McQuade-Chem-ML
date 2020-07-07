from py2neo import Graph
from py2neo.database import ClientError
import json


def __check_for_constraint__(graph):
    """
    This function will check and make sure that constraints are on the graph. This is important because
    constraints allow nodes to replace IDs with, well the constraint label. This allows Neo4j to quickly merge
    the different nodes. Constraints plus UNWIND allow for some really impressive speed up times in inserting
    data into Neo4j.
    """

    constraint_check_string = """
    
    CREATE CONSTRAINT unique_Algorithm_name
    ON (n:Algorthim)
    ASSERT n.name IS UNIQUE
    
    CREATE CONSTRAINT unique_TaskType_type
    ON (n:TaskType)
    ASSERT n.type IS UNIQUE
    
    CREATE CONSTRAINT unique_Dataset_name
    ON (n:Dataset)
    ASSERT n.name IS UNIQUE
    
    CREATE CONSTRAINT unique_FeaturizeMethod_feat_ID
    ON (n:FeaturizeMethod)
    ASSERT n.feat_ID IS UNIQUE
    
    CREATE CONSTRAINT unique_Model_run_name
    ON (n:Model)
    ASSERT n.run_name IS UNIQUE
    
    """

    try:
        tx = graph.begin(autocommit=True)
        tx.evaluate(constraint_check_string)
    except ClientError:
        pass


def __get_model_query__():
    query = """
    
    UNWIND $run as run
    MERGE (algo:Algorithm {name: run.algorithm})
    MERGE (task:TaskType {type: run.task_type})
    
    MERGE (dataset:Dataset {name: run.dataset})
    ON CREATE SET dataset.target_name = run.target_name
    
    MERGE (feat_method:FeaturizeMethod {feat_ID: run.feat_meth})
    FOREACH (feat_name in run.feature_list | 
                 MERGE (feat:Feature {name: feat_name}) 
                 MERGE (feat_method)-[:use_feature]->(feat)
                )
                
    MERGE (model:Model {run_name: run.run_name})
    ON CREATE SET model.random_seed = run.random_seed, model.opt_iter = model.opt_iter,
                  model.cv_folds = run.cv_folds, model.tuned = run.tuned, model.tune_time = run.tune_time,
                  model.feat_time = run.feat_time, model.test_percent = run.test_percent,
                  model.val_percent = run.val_percent, model.train_percent = run.train_percent,
                  model.n_tot = run.n_tot, model.in_shape = run.in_shape, model.n_val = run.n_val,
                  model.n_train = run.n_train, model.n_test = run.n_test,
                  model.r2_raw = run.r2_raw, model.r2_avg = run.r2_avg, model.r2_std = run.r2_std,
                  model.mse_raw = run.mse_raw, model.mse_avg = run.mse_avg, model.mse_std = run.mse_std,
                  model.rmse_raw = run.rmse_raw, model.rmse_avg = run.rmse_avg, model.rmse_std = run.rmse_std,
                  model.time_raw = run.time_raw, model.time_avg = run.time_avg, model.time_std = run.time_std
                  
    MERGE (model)-[:uses_featurize_method]->(feat_method)
    MERGE (model)-[:uses_algorithm]->(algo)
    MERGE (model)-[:uses_task]->(task)
    """
    return query


def model_json_to_neo4j(json_file):
    graph = Graph()
    __check_for_constraint__(graph)

    with open(json_file, 'r') as f:
        json_file_data = f.read()
    data = json.loads(json_file_data)

    for label, value in data['predictions_stats'].items():
        data[label] = value
    data.pop('predictions_stats')

    tx = graph.begin(autocommit=True)
    tx.evaluate(__get_model_query__(), parameters={"run": data})


if __name__ == "__main__":
    model_json_to_neo4j('/home/user/PycharmProjects/McQuade-Chem-ML/output/Aw00_20200706-123430/Aw00_20200706-123430_attributes.json')
