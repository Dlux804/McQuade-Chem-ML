"""
Objective: Create Neo4j graphs from files inside of "output"
"""
import pandas as pd
import os
from core import json_cypher, split_smiles_to_neo4j, features_to_neo4j
from py2neo import Graph
from zipfile import ZipFile
from core import misc
import json

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def dataframe_section(dataframe, split):
    """
    Create a smaller dataframe based on their split
    :param dataframe: _data.csv dataframe
    :param value: a string that is either: train, test or val
    :return: A smaller dataframe that only contains the "split" string on its "in_set" column
    """
    split = ''.join([split[0].lower(), split[1:]])
    df = dataframe.loc[dataframe['in_set'] == split]
    df = df.reset_index(drop=True)
    df = df.drop(['Unnamed: 0'], axis=1)
    return df


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


def output_to_neo4j():
    """"""
    json_generator = get_file('_attributes.json')
    csv_generator = get_file('_data.csv')
    for i in range(file_count()):
        # Json dataframe
        df_from_attributes = next(json_generator)
        # pandas dataframe
        df_from_data = next(csv_generator)
        data_list = list(df_from_attributes.T.to_dict().values())  # List of dictionary from json attributes
        json_cypher.non_loop_graph(data_list, df_from_attributes)
        json_cypher.if_nodes(data_list, df_from_attributes)
        json_cypher.feature_method_node(df_from_attributes)
        val_percent = float(df_from_attributes['val_percent'])
        train_test_val = ['Train', 'Test', 'Val']
        if val_percent > 0.0:
            for split in train_test_val:
                split_df = dataframe_section(df_from_data, split)
                split_smiles_to_neo4j.split_to_neo(split_df, df_from_attributes, data_list, split)
        else:
            train_test_val.remove('Val')
            for split in train_test_val:
                split_df = dataframe_section(df_from_data, split)
                split_smiles_to_neo4j.split_to_neo(split_df, df_from_attributes, data_list, split)
        features_to_neo4j(df_from_attributes, df_from_attributes)

# print(df_from_attributes)
output_to_neo4j()



