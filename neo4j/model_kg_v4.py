import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


def query_line(label, name, target):
    query = r'''match (n:{0}) where n.{1} = '{2}' with collect (n) 
            as ns call apoc.refactor.mergeNodes(ns) yield node return node;'''.format(label, name, target)
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
    def get_unique(cls, df, col, num_data=None):
        """
        Objective: Get unique elements in a csv column so we can use them as the main nodes.
        :param df:
        :param col:
        :param num_data:
        :return: unique elements in a pandas series (csv column)
        """

        data_lst = df[col].tolist()
        unique = []
        for i in data_lst:
            if i not in unique:
                unique.append(i)
        print(unique)
        if num_data is None:
            print('   {:5}    {:>15}'.format("Selection", "Options"))
            [print('{:^15} {}'.format(*data)) for data in enumerate(unique)]
            num_data = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
        selected = [unique[i] for i in num_data]
        print("You have selected the following: ", end="   ", flush=True)
        print(*selected, sep=', ')
        return selected

    @staticmethod
    def query(col):
        """

        :param col: csv column name
        :return: nodes' labels and names
        """
        label_grid = {
            "algorithm": ['algo', 'algorithm'],
            "dataset": ['data_ml', 'data'],
            "feat_meth": ['featmeth', 'feat_meth'],
            "target": ['targets', 'target'],
            "tuned": ['tuned', 'tuned']
        }
        return label_grid[col]

    def nodes_relationships(self):
        """
        This staticmethod will create nodes and relationships based on the available data after different
        functions have been run based on the questions asked

        :return: Nodes and Relationships in Neo4j Desktop
        :prerequisite: have Neo4j Desktop opened
        """
        df = self.data
        graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
        model_dicts = df.to_dict('records')
        regressor_lst = []
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
            feature_list = Node("featurelist", feature_lists=ml_dict['feature_list'])
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
            tx.commit()
            regressor_lst.append(regressor)
        return regressor_lst


def get_query(df, col, num_data=None):
    """
    Objective: Return a list of cypher commands to run them on Neo4j Desktop.
    :param df: pandas dataframe
    :param col: csv column name
    :param num_data: list of wanted elements in terms of numbers
    :return: a list of cypher commands
    """
    select = ML_kg.get_unique(df, col, num_data)
    # print(select)
    unique = ML_kg.query(col)
    query_lst = []
    for i in select:
        query = query_line(unique[0], unique[1], i)
        query_lst.append(query)
    return query_lst


def run_cypher_command(file, col):
    """
    Objective: Automate the task of running Cypher commands in Neo4j Desktop
    :param file: csv file
    :param col: csv column name
    :return: Run cypher commands in Neo4j Desktop
    """
    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    t = ML_kg(file)
    queries = get_query(t.data, col)
    for queue in queries:
        graph.evaluate(queue)


t = ML_kg('ml_results2.csv')
# t.nodes_relationships()
# run_cypher_command(t.file, "target")
run_cypher_command(t.file, "algorithm")
# run_cypher_command(t.file, "dataset")
# run_cypher_command(t.file, "algorithm")
# run_cypher_command(t.file, "tuned")
# queries = get_query(t.data, "feat_meth")
# for queue in queries:
#     graph.evaluate(queue)


# a = t.query("dataset")
# for i in selected:
#     query = test_apoc(a[0], a[1], i)
# print(a[0])
# print(query)
