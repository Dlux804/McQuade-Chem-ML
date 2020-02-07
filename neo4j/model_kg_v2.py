import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


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
            print('Creating Relationships Number ' + str(i))
            ab = Relationship(data, "uses", algo)
            tx.merge(ab)
            ac = Relationship(data, "experimental", target)
            tx.merge(ac)
            ad = Relationship(data, "generates", feat_meth)
            tx.merge(ad)
            ae = Relationship(feat_meth, "feat_time", feat_time)
            tx.merge(ae)
            af = Relationship(feat_meth, "means", feature_list)
            tx.merge(af)
            ag = Relationship(data, "tuned", tuned)
            tx.merge(ag)
            ah = Relationship(tuned, "params", regressor)
            tx.merge(ah)
            ai = Relationship(tuned, "tunetime", tunetime)
            tx.merge(ai)
            aj = Relationship(regressor, "r2_std", r2_std)
            tx.merge(aj)
            ak = Relationship(regressor, "r2_avg", r2_avg)
            tx.merge(ak)
            al = Relationship(regressor, "mse_std", mse_std)
            tx.merge(al)
            am = Relationship(regressor, "mse_avg", mse_avg)
            tx.merge(am)
            an = Relationship(regressor, "rmse_std", rmse_std)
            tx.merge(an)
            ao = Relationship(regressor, "rmse_avg", rmse_avg)
            tx.merge(ao)
            ap = Relationship(data, "time_std", time_std)
            tx.merge(ap)
            aq = Relationship(data, "time_avg", time_avg)
            tx.merge(aq)
            tx.commit()

    def high_r2avg(self):
        """
        Objective: Return dataframe with runs that have high (>0.88) r2_avg value
        :return: Nodes and Relationships in Neo4j
        """
        df = self.data
        final = df[df.r2_avg.mean() >= 0.88]
        ML_kg.nodes_relationships(final)
        # final_df = df.drop()

    def low_mseavg(self):
        """
        Objective: Return dataframe with runs that have low (<0.4) mse_avg value
        :return: Nodes and Relationships in Neo4j
        """
        df = self.data
        final = df[df.mse_avg <= 0.4]
        ML_kg.nodes_relationships(final)

    def low_rmseavg(self):
        """
        Objective: Return dataframe with runs that have low (<0.4) rmse_avg value
        :return:Nodes and Relationships in Neo4j
        """
        df = self.data
        final = df[df.mse_avg <= 0.65]
        ML_kg.nodes_relationships(final)

    def avg_r2avg(self):
        """
        Objective: Return dataframe with runs that have r2 avg lower than the mean
        :return:
        """
        df = self.data
        col_mean = df['r2_avg'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.r2_avg < col_mean]
        ML_kg.nodes_relationships(final)

    def avg_mseavg(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['mse_avg'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.mse_avg > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def avg_rmseavg(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['rmse_avg'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.rmse_avg > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def high_r2std(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['r2_std'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.r2_std > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def high_msestd(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['mse_std'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.mse_std > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def high_rmsestd(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['rmse_std'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.rmse_std > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def high_tunetime(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['tuneTime'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.tuneTime > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)

    def high_fittimeavg(self):
        """

        :return:
        """
        df = self.data
        col_mean = df['time_avg'].mean()
        print('The mean of this column is ' + str(col_mean))
        final = df[df.time_avg > col_mean]
        print(final)
        ML_kg.nodes_relationships(final)


ML_kg('merged_MLoutput.csv').high_rmsestd()
