import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher

"""
    Main goal: Make cypher commands that would merge duplicate nodes together.
    Though process: Get all unique values in a column, get names and labels of the nodes, then execute query lines in 
                    Cypher. 
"""


def query_line(label, name, target):
    """
    Objective: Create query that is formatted in a way that neo4j can read and execute
    :param label: Name of the node's label
    :param name: Name of node
    :param target: node's id
    :return: query: return query line
    """
    query = r'''match (n:{0}) where n.{1} = '{2}' with collect (n) 
            as ns call apoc.refactor.mergeNodes(ns) yield node return node;'''.format(label, name, target)
    return query


class Cypher:
    """
    Objective: Make different graphs that enable users to answer different questions based on the data presented.
    Run these 2 commands on Cypher to merge algorithms

    """

    @classmethod
    def get_unique(cls, df, col, num_data=None):
        """
        Objective: Get unique elements in csv so we can merge them in neo4j.
        :param df: dataframe
        :param col: column name
        :param num_data: Number of things we want to merge
        :return: selected: unique elements in a pandas series (csv column)
        """

        data_lst = df[col].tolist()  # Get column values
        unique = []  # Unique list
        for i in data_lst:  # Enumerate over all values
            if i not in unique: # If value is not on the list (unique) then append them to list
                unique.append(i)
        print(unique)
        if num_data is None:  # If nothing is specifies
            selected = unique  # Then select everything on the list
        # selected = [unique[i] for i in num_data]
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

    @staticmethod
    def get_query(df, col, num_data=None):
        """
        Objective: Return a list of cypher commands to run them on Neo4j Desktop.
        :param df: pandas dataframe
        :param col: csv column name
        :param num_data: list of wanted elements in terms of numbers
        :return: a list of cypher commands
        """
        select = Cypher.get_unique(df, col, num_data)  # Get list of unique values
        # print(select)
        unique = Cypher.query(col)  # Get labels and names of nodes
        query_lst = []
        for i in select:  # Enumerate over unqiue values
            query = query_line(unique[0], unique[1], i)  # Get query lines
            query_lst.append(query)  # List of all query lines
        return query_lst

    def cypher_command(self, df, col):
        """
        Objective: Automate the task of running Cypher commands in Neo4j Desktop
        :param df: csv file
        :param col: csv column name
        :return: Run cypher commands in Neo4j Desktop
        """
        graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")  # connect to neo4j
        queries = Cypher.get_query(df, col)  # Get list of all query lines
        for queue in queries:
            graph.evaluate(queue)  # Execute every query line

# run_cypher_command(t.file, "target")
# run_cypher_command(t.file, "algorithm")
# run_cypher_command(pd.read_csv('ml_results2.csv'), "dataset")
# run_cypher_command(t.file, "algorithm")
# run_cypher_command(t.file, "tuned")
