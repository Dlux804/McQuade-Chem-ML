"""
Objective: Create Neo4j graphs from files inside of "output"
"""
import pandas as pd
import os
from py2neo import Graph
from zipfile import ZipFile
from core import misc
import json
from core import data_nodes_to_neo4j, data_rel_to_neo4j, prep_from_outputs

# TODO: Add documentation

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def file_count():
    """

    :return:
    """
    # print(hello)
    with misc.cd('../output/'):  # Access output
        print('Now in:', os.getcwd())
        for roots, dirs, files in os.walk(os.getcwd()):
            file_count = 0
            json_list = []
            csv_list = []
            for f in files:
                if f.endswith('.zip'):
                    file_count += 1
            return file_count


def get_file(file_string):
    """"""
    with misc.cd('../output/'):  # Access output
        for roots, dirs, files in os.walk(os.getcwd()):
            for f in files:
                if f.endswith('.zip'):
                    with ZipFile(f, 'r') as zip:
                        file_list = zip.namelist()
                        for file in file_list:
                            if file_string in file and file_string == "_attributes.json":
                                with zip.open(file) as json_file:  # Open file in zip
                                    read_json = json_file.read()  # Load in read function
                                    data = json.loads(read_json.decode("utf-8"))  # Decode json before loading
                                    df_from_attributes = pd.json_normalize(data)  # Normalizing json data into dataframe
                                yield df_from_attributes
                            if file_string in file and file_string == "_data.csv":
                                df_from_data = pd.read_csv(zip.open(file))
                                yield df_from_data
                            if file_string in file and file_string == "_predictions.csv":
                                df_from_predictions = pd.read_csv(zip.open(file))
                                yield df_from_predictions


def output_to_neo4j():
    """"""
    json_generator = get_file('_attributes.json')
    data_csv_generator = get_file('_data.csv')
    predictions_csv_generator = get_file("_predictions.csv")
    for i in range(file_count()):
        # Json dataframe
        df_from_attributes = next(json_generator)
        # pandas dataframe
        df_from_data = next(data_csv_generator)
        df_from_predictions = next(predictions_csv_generator)
        prep = prep_from_outputs.Prep(df_from_attributes, df_from_predictions, df_from_data)
        # print(prep.n_test)
        data_nodes_to_neo4j.nodes(prep)
        data_rel_to_neo4j.relationships(prep)

# output_to_neo4j()



