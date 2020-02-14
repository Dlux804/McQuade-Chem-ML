import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher

def test_apoc():
    query = r'''match (n:algo) where n.algorithm = '{}' with collect (n) 
            as ns call apoc.refactor.mergeNodes(ns) yield node return node;'''.format('gdb')
    return query

class ML_kg:
    """
    Objective: Make different graphs that enable users to answer different questions based on the data presented.
    Run these 2 commands on Cypher to merge algorithms
        MATCH (n:algo) WHERE n.algorithm = 'gdb' WITH COLLECT (n) AS ns CALL apoc.refactor.mergeNodes(ns) YIELD node RETURN node;
        MATCH (n:algo) WHERE n.algorithm = 'rf' WITH COLLECT (n) AS ns CALL apoc.refactor.mergeNodes(ns) YIELD node RETURN node;
    """

    def __init__(self, file):
        self.file = file
        self.data = pd.read_csv(self.file)

    @classmethod
    def nodes_relationships(cls, final_df):
        """
        This classmethod will create nodes and relationships based on the available data after different
        functions have been run based on the questions asked

        :param final_df: final dataframe
        :return: Nodes and Relationships in Neo4j Desktop
        :prerequisite: have Neo4j Desktop opened
        """
        df = final_df
        graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
        tx = graph.begin()
        model_dicts = df.to_dict('records')
        for i in range(len(model_dicts)):
            ml_dict = model_dicts[i]
            print('Creating Nodes Number ' + str(i))
            tx = graph.begin()
            runs = Node("run_num", run=ml_dict['Run#'])
            tx.create(runs)
            algo = Node("algo", algorithm=ml_dict['algorithm'])
            tx.create(algo)
            data = Node("data_ml", data=ml_dict['dataset'])
            tx.create(data)
            target = Node("targets", target=ml_dict['target'])
            tx.create(target)
            feat_meth = Node("featmeth", feat_meth=ml_dict['feat_meth'])
            tx.create(feat_meth)
            feat_time = Node("feattime", feat_time=ml_dict['feat_time'])
            tx.create(feat_time)
            tuned = Node("tuned", tuned=ml_dict['tuned'])
            tx.create(tuned)
            feature_list = Node("featurelist", feature_list=ml_dict['feature_list'])
            tx.create(feature_list)
            regressor = Node("regress", regressor=ml_dict['regressor'])
            tx.create(regressor)
            tunetime = Node("tunetimes", tunetime=ml_dict['tuneTime'])
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
            final_results = Node("results", result=ml_dict['Results'])
            tx.create(final_results)
            print('Creating Relationships Number ' + str(i))
            aa = Relationship(runs, "uses", algo)
            tx.merge(aa)
            ab = Relationship(runs, "uses", data)
            tx.merge(ab)
            ac = Relationship(data, "has", target)
            tx.merge(ac)
            ad = Relationship(runs, "generates", feat_meth)
            tx.merge(ad)
            ae = Relationship(feature_list, "feat_time", feat_time)
            tx.merge(ae)
            af = Relationship(feat_meth, "means", feature_list)
            tx.merge(af)
            ag = Relationship(tuned, "tuned", regressor)
            tx.merge(ag)
            ah = Relationship(algo, "params", regressor)
            tx.merge(ah)
            ai = Relationship(tuned, "tunetime", tunetime)
            tx.merge(ai)
            aj = Relationship(regressor, "gives", final_results)
            tx.merge(aj)
            ak = Relationship(final_results, "has r2_avg", r2_avg)
            tx.merge(ak)
            al = Relationship(final_results, "has mse_std", mse_std)
            tx.merge(al)
            am = Relationship(final_results, "has mse_avg", mse_avg)
            tx.merge(am)
            an = Relationship(final_results, "has Rmse_std", rmse_std)
            tx.merge(an)
            ao = Relationship(final_results, "has Rmse_avg", rmse_avg)
            tx.merge(ao)
            ap = Relationship(final_results, "has r2_std", r2_std)
            tx.merge(ap)
            aq = Relationship(regressor, "has time_std", time_std)
            tx.merge(aq)
            ar = Relationship(regressor, "has time_avg", time_avg)
            tx.merge(ar)
            at = Relationship(algo, "tune", tuned)
            tx.merge(at)
            au = Relationship(runs, "gives", final_results)
            tx.merge(au)
            av = Relationship(algo, "contributes to", final_results)
            tx.merge(av)
            az = Relationship(data, "contributes to", final_results)
            tx.merge(az)
            bb = Relationship(feat_meth, "contributes to", final_results)
            tx.merge(bb)
            tx.evaluate(test_apoc())
            tx.commit()

    def esol_model(self):
        df = self.data
        col = df[df.dataset == 'ESOL.csv']
        ML_kg.nodes_relationships(col)
        print('complete')

    def lipo_model(self):
        df = self.data
        col = df[df.dataset == "Lipophilicity-ID.csv"]
        ML_kg.nodes_relationships(col)
        print('complete')

    def water_en_model(self):
        df = self.data
        col = df[df.dataset == "water-energy.csv"]
        ML_kg.nodes_relationships(col)
        print('complete')

    def jak2(self):
        df = self.data
        col = df[df.dataset == "jak2_pic50.csv"]
        ML_kg.nodes_relationships(col)
        print('complete')


model = ML_kg('ml_results2.csv')
model.jak2()

# def nodes_relationships(self, final_df):
#     """
#     This staticmethod will create nodes and relationships based on the available data after different
#     functions have been run based on the questions asked
#
#     :param final_df: final dataframe
#     :return: Nodes and Relationships in Neo4j Desktop
#     :prerequisite: have Neo4j Desktop opened
#     """
#     df = self.data
#     graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
#     tx = graph.begin()
#     model_dicts = df.to_dict('records')
#     for i in range(len(model_dicts)):
#         ml_dict = model_dicts[i]
#         print('Creating Nodes Number ' + str(i))
#         tx = graph.begin()
#         runs = Node("run_num", run=ml_dict['Run#'])
#         tx.create(runs)
#         algo = Node("algo", algorithm=ml_dict['algorithm'])
#         tx.create(algo)
#         data = Node("data_ml", data=ml_dict['dataset'])
#         tx.create(data)
#         target = Node("targets", target=ml_dict['target'])
#         tx.create(target)
#         feat_meth = Node("featmeth", feat_meth=ml_dict['feat_meth'])
#         tx.create(feat_meth)
#         feat_time = Node("feattime", feat_time=ml_dict['feat_time'])
#         tx.create(feat_time)
#         tuned = Node("tuned", tuned=ml_dict['tuned'])
#         tx.create(tuned)
#         feature_list = Node("featurelist", feature_list=ml_dict['feature_list'])
#         tx.create(feature_list)
#         regressor = Node("regress", regressor=ml_dict['regressor'])
#         tx.create(regressor)
#         tunetime = Node("tunetimes", tunetime=ml_dict['tuneTime'])
#         tx.create(tunetime)
#         r2_avg = Node("r2avg", r2_avg=ml_dict['r2_avg'])
#         tx.create(r2_avg)
#         r2_std = Node("r2std", r2_std=ml_dict['r2_std'])
#         tx.create(r2_std)
#         mse_avg = Node("mseavg", mse_avg=ml_dict['mse_avg'])
#         tx.create(mse_avg)
#         mse_std = Node("msestd", mse_std=ml_dict['mse_std'])
#         tx.create(mse_std)
#         rmse_avg = Node("rmseavg", rmse_avg=ml_dict['rmse_avg'])
#         tx.create(rmse_avg)
#         rmse_std = Node("rmsestd", rmse_std=ml_dict['rmse_std'])
#         tx.create(rmse_std)
#         time_avg = Node("timeavg", time_avg=ml_dict['time_avg'])
#         tx.create(time_avg)
#         time_std = Node("timestd", time_std=ml_dict['time_std'])
#         tx.create(time_std)
#         final_results = Node("results", result=ml_dict['Results'])
#         tx.create(final_results)
#         print('Creating Relationships Number ' + str(i))
#         aa = Relationship(runs, "uses", algo)
#         tx.merge(aa)
#         ab = Relationship(runs, "uses", data)
#         tx.merge(ab)
#         ac = Relationship(data, "has", target)
#         tx.merge(ac)
#         ad = Relationship(runs, "generates", feat_meth)
#         tx.merge(ad)
#         ae = Relationship(feature_list, "feat_time", feat_time)
#         tx.merge(ae)
#         af = Relationship(feat_meth, "means", feature_list)
#         tx.merge(af)
#         ag = Relationship(tuned, "tuned", regressor)
#         tx.merge(ag)
#         ah = Relationship(algo, "params", regressor)
#         tx.merge(ah)
#         ai = Relationship(tuned, "tunetime", tunetime)
#         tx.merge(ai)
#         aj = Relationship(regressor, "gives", final_results)
#         tx.merge(aj)
#         ak = Relationship(final_results, "has r2_avg", r2_avg)
#         tx.merge(ak)
#         al = Relationship(final_results, "has mse_std", mse_std)
#         tx.merge(al)
#         am = Relationship(final_results, "has mse_avg", mse_avg)
#         tx.merge(am)
#         an = Relationship(final_results, "has Rmse_std", rmse_std)
#         tx.merge(an)
#         ao = Relationship(final_results, "has Rmse_avg", rmse_avg)
#         tx.merge(ao)
#         ap = Relationship(final_results, "has r2_std", r2_std)
#         tx.merge(ap)
#         aq = Relationship(regressor, "has time_std", time_std)
#         tx.merge(aq)
#         ar = Relationship(regressor, "has time_avg", time_avg)
#         tx.merge(ar)
#         at = Relationship(algo, "tune", tuned)
#         tx.merge(at)
#         au = Relationship(runs, "gives", final_results)
#         tx.merge(au)
#         av = Relationship(algo, "contributes to", final_results)
#         tx.merge(av)
#         az = Relationship(data, "contributes to", final_results)
#         tx.merge(az)
#         bb = Relationship(feat_meth, "contributes to", final_results)
#         tx.merge(bb)
#         tx.evaluate(lambda file, col: get_query(file, col))
#         tx.evaluate()
#         tx.commit()