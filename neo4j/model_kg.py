import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


# class ML_kg:
#     """
#
#     """
#     def __init__(self, file):
#         self.file = file
#
#     def insert_ml_runs(self):
#         self.graph = Graph("bolt://localhost:7687", user="neo4j", password="123")
#         ml_data = pd.read_csv(self.file)
#         ml_dicts = ml_data.to_dict('records')
#         for i in range(len(ml_dicts)):
#             ml_dict = ml_dicts[i]
#             ml_model = Node("model", algorithm=ml_dict['algorithm'], data=ml_dict['dataset'], target=ml_dict['target'],
#                         feat_meth=ml_dict['feat_meth'], feat_time=ml_dict['feat_time'], tuned=ml_dict['tuned'],
#                         feature_list=ml_dict['feature_list'], regressor=ml_dict['regressor'],
#                         tunetime=ml_dict['tuneTime'], r2_avg=ml_dict['r2_avg'], r2_std=ml_dict['r2_std'],
#                         mse_avg=ml_dict['mse_avg'], mse_std=ml_dict['mse_std'], rmse_avg=ml_dict['rmse_avg'],
#                         rmse_std=ml_dict['rmse_std'], time_avg=ml_dict['time_avg'], time_std=ml_dict['time_std'])
#             algo = Node("model", algorithm=ml_dict['algorithm'])
#             data = Node("model", data=ml_dict['dataset'])
#             target =
#             graph.create(ml_model)
#
#
#     @classmethod
#     def get_relationship(cls):
#         cls.insert_ml_runs(cls.file)


graph = Graph("bolt://localhost:7687", user="neo4j", password="123")
ml_data = pd.read_csv('merged_MLoutput.csv')
ml_dicts = ml_data.to_dict('records')
for i in range(len(ml_dicts)):
    ml_dict = ml_dicts[i]
    tx = graph.begin()
    algo = Node("algo", algorithm=ml_dict['algorithm'])
    tx.create(algo)
    data = Node("data", data=ml_dict['dataset'])
    tx.create(data)
    target = Node("target", target=ml_dict['target'])
    tx.create(target)
    feat_meth = Node("featmeth", feat_meth=ml_dict['feat_meth'])
    tx.create(feat_meth)
    feat_time = Node("feattime", feat_time=ml_dict['feat_time'])
    tx.create(feat_time)
    tuned = Node("tuned", tuned=ml_dict['tuned'])
    tx.create(tuned)
    feature_list = Node("featurelist", feature_list=ml_dict['feature_list'])
    tx.create(feature_list)
    regressor = Node("regressor", regressor=ml_dict['regressor'])
    tx.create(regressor)
    tunetime = Node("tunetime", tunetime=ml_dict['tuneTime'])
    tx.create(tunetime)
    r2_avg = Node("r2avg", r2_avg=ml_dict['r2_avg'])
    tx.create(r2_avg)
    r2_std = Node("r2std", r2_std=ml_dict['r2_std'])
    tx.create(r2_std)
    mse_avg = Node("mseavg", mse_avg=ml_dict['mse_avg'])
    tx.create(mse_avg)
    mse_std = Node("msestd", mse_std=ml_dict['mse_std'])
    tx.create(mse_std)
    rmse_avg = Node("rmseavg", rmse_avg=ml_dict['rmse_avg'])
    tx.create(rmse_avg)
    rmse_std = Node("rmsestd", rmse_std=ml_dict['rmse_std'])
    tx.create(rmse_std)
    time_avg = Node("timeavg", time_avg=ml_dict['time_avg'])
    tx.create(time_avg)
    time_std = Node("timestd", time_std=ml_dict['time_std'])
    tx.create(time_std)
    ab = Relationship(data, "uses", algo)
    tx.merge(ab)
    ac = Relationship(data, "contains", target)
    tx.merge(ac)

    tx.commit()
