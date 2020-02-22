import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


def query_line(label, name, target):
    query = r'''match (n:{0}) where n.{1} = '{2}' with collect (n) 
            as ns call apoc.refactor.mergeNodes(ns) yield node return node;'''.format(label, name, target)
    return query


class Cypher:
    """
    Objective: Make different graphs that enable users to answer different questions based on the data presented.
    Run these 2 commands on Cypher to merge algorithms

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
            "tuned": ['tuned', 'tune'],
            "regressor": ['regress', 'regressor']
        }
        return label_grid[col]



def get_query(df, col, num_data=None):
    """
    Objective: Return a list of cypher commands to run them on Neo4j Desktop.
    :param df: pandas dataframe
    :param col: csv column name
    :param num_data: list of wanted elements in terms of numbers
    :return: a list of cypher commands
    """
    select = Cypher.get_unique(df, col, num_data)
    # print(select)
    unique = Cypher.query(col)
    query_lst = []
    for i in select:
        query = query_line(unique[0], unique[1], i)
        query_lst.append(query)
    return query_lst


def run_cypher_command(df, col):
    """
    Objective: Automate the task of running Cypher commands in Neo4j Desktop
    :param file: csv file
    :param col: csv column name
    :return: Run cypher commands in Neo4j Desktop
    """
    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    queries = get_query(df, col)
    for queue in queries:
        graph.evaluate(queue)



# run_cypher_command(t.file, "target")
# run_cypher_command(t.file, "algorithm")
# run_cypher_command(pd.read_csv('ml_results2.csv'), "dataset")
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
