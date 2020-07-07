"""
Objective: Create Neo4j graphs from files inside of "output"
"""
import pandas as pd
import os
import pathlib
from py2neo import Graph
from zipfile import ZipFile
from core import misc
import json

query_line = """ 
    UNWIND $parameters as rows
    merge (data_algor:data_algorithms {data_algorithm: rows.algorithm})
"""

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


def value_to_list(df, headers):
    """
    Put every value in the dataframe into a list
    :param df:
    :param headers:
    :return:
    """
    col_list = []
    for col in headers:
        column = df[col].tolist()
        new_col = [[col] for col in column]
        col_list.append(new_col)
    rotate_list = list(rotated(col_list))
    df = pd.DataFrame.from_records(rotate_list, columns=headers)
    return df


def get_batches(lst, batch_size=100):
    return [(i, lst[i:i + batch_size]) for i in range(0, len(lst), batch_size)]


def run_neo_query(data, query):
    batches = get_batches(data)

    for index, batch in batches:
        print('[Batch: %s] Will add %s node to Graph' % (index, len(batch)))
        g.run(query, rows=batch, parameters={'parameters': data})


def access_ouput():
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
                        csv_str = "_predictions.csv"
                        for file in file_list:
                            if json_str in file:
                                with zip.open(file) as json_file:  # Open file in zip
                                    read_json = json_file.read()  # Load in read function
                                    data = json.loads(read_json.decode("utf-8"))  # Decode json before loading
                                    df_normalize = pd.json_normalize(data)  # Normalizing json data into dataframe
                                    # df_data = df_normalize[['algorithm', 'dataset']]
                                    data_list = list(df_normalize.T.to_dict().values())
                                    # tx = g.begin()
                                    g.evaluate(query_line, parameters={'parameters': data_list})

                            #         header = df_normalize.columns.tolist()
                            #         df_list = value_to_list(df_normalize, header)
                            #         print(df_list)
                            #         parameter = []
                            #         for index, row in df_list.iterrows():
                            #             # print("Row number:", index)
                            #             # parameter.append(dict(row))
                            #             tx = g.begin()
                            #             tx.evaluate(query_line)
                            #             # if index % 20000 == 0 and index > 0:
                            # if csv_str in file:
                            #     with zip.open(file) as csv_file:
                            #         df = pd.read_csv(csv_file)
                            #         df_data = df[['algorithm', 'dataset']]
                            #         print(df_data)
                                    # header = df.columns.tolist()
                                    # df_list = value_to_list(df, header)
                                    # print(df_list)
                                    # parameter = []
                                    # for index, row in df_list.iterrows():
                                    #     # print("Row number:", index)
                                    #     parameter.append(dict(row))
                                    #
                                    #     tx = g.begin(autocommit=True)
                                    #     tx.evaluate(query_line)
                                #     df = pd.DataFrame(d)
                                # print(df)


access_ouput()