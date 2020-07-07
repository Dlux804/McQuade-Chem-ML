"""
Objective: Create Neo4j graphs from files inside of "output"
"""
import pandas as pd
import os
from core import json_cypher
from py2neo import Graph
from zipfile import ZipFile
from core import misc
import json


g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def attributes_neo4j():
    """

    :return:
    """
    with misc.cd('../output/'):  # Access output
        print('Now in:', os.getcwd())
        for roots, dirs, files in os.walk(os.getcwd()):
            for f in files:
                if f.endswith('.zip'):
                    print("working with:", f)
                    with ZipFile(f, 'r') as zip:
                        file_list = zip.namelist()
                        json_str = "_attributes.json"
                        for file in file_list:
                            if json_str in file:
                                with zip.open(file) as json_file:  # Open file in zip
                                    read_json = json_file.read()  # Load in read function
                                    data = json.loads(read_json.decode("utf-8"))  # Decode json before loading
                                    df_normalize = pd.json_normalize(data)  # Normalizing json data into dataframe
                                    data_list = list(df_normalize.T.to_dict().values())
                                    # feature length
                                    feature_length = len(df_normalize['feature_list'])
                                    # test time
                                    test_time = df_normalize["predictions_stats.time_avg"]
                                    # train size
                                    train_size = len(df_normalize['target_array']) * df_normalize['train_percent']
                                    # test size
                                    test_size = len(df_normalize['target_array']) * df_normalize['test_percent']
                                    algorithm = str(df_normalize['algorithm'])  # algorithm
                                    # validation size
                                    val_size = len(df_normalize['target_array']) * df_normalize['val_percent']
                                    tuned = str(df_normalize['tuned'])  # tuned or not
                                    feat_name = list(df_normalize['feat_name'])  # feature method
                                    # train_molecules = list(df_normalize['train_molcules'])
                                    # test_molecules = list(df_normalize['test_molecules'])
                                    # smiles_list = list(self.data['smiles'])  # List of all SMILES
                                    json_cypher.non_loop_graph(data_list, feature_length, test_time, train_size,
                                                               test_size)
                                    json_cypher.if_nodes(data_list, algorithm, tuned, float(val_size))
                                    json_cypher.feature_method_node(feat_name, feature_length)
                                    json_cypher.train_test_molecules(train_molecules, test_molecules)


attributes_neo4j()



