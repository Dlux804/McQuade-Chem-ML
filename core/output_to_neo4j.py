"""
Objective: Contains the function "output_to_neo4j" which is the function that imports data from output folders into
            Neo4j based in our ontology
"""
import pandas as pd
import os
from py2neo import Graph
from zipfile import ZipFile
from core import misc
import json
from core import prep_from_outputs


g = Graph("neo4j:1234@localhost:7687/db/data/", bolt=True)


def file_count():
    """
    Objective: Count the number of zip files in the "output" folder.
    Intent: I want to use the "yield" command to return dataframe in a for loop later on, which will create generator
            objects. Having the file counts me the ability to move to the next variable stored in the generator object.
    More info: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    :return: Number of zip files in the output folder
    """
    # print(hello)
    with misc.cd('../output/'):  # Access output
        print('Now in:', os.getcwd())
        for roots, dirs, files in os.walk(os.getcwd()):
            files_count = 0
            for f in files:
                if f.endswith('.zip'):
                    files_count += 1
                if f.endswith('_qdb.zip'):  # Ignore qsardb zip files
                    pass
            print(f"You have {files_count} zip files in your output folder")
            return files_count


def get_file(file_string):
    """
    Objective: Yield dataframes in a for loop for memory efficiency.
    Intent: I want this function to access every zip files in the "output" folder and turn the necessary files into
            dataframe. The key here is to do it as fast and memory efficient as possible. The Yield command ... yields
            a generator object that is much lighter by nature than other storage data type like lists or tuples. It will
            bloat up memory if I return a list or a tuple of 100s of dataframes and bottleneck the process.
    :param file_string: A common substring in a file name in our zip files
    :return: 3 dataframe from _attributes.json, _data.csv and _predictions.csv
    """
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
                                df_from_data = pd.read_csv(zip.open(file))  # Open dataframe in zip
                                yield df_from_data
                            if file_string in file and file_string == "_predictions.csv":
                                df_from_predictions = pd.read_csv(zip.open(file))  # Open dataframe in zip
                                yield df_from_predictions


def output_to_neo4j():
    """
    Objective: Import data from output files into Neo4j based on our ontology.
    Intent: This want this function to be the final command that will put data into Neo4j. Should not contain anything
            that doesn't help accomplish that goal.
    :return:
    """
    json_generator = get_file('_attributes.json')  # attribute file string
    data_csv_generator = get_file('_data.csv')  # data file string
    predictions_csv_generator = get_file("_predictions.csv")  # predictions file string
    for i in range(file_count()):
        # Json dataframe
        df_from_attributes = next(json_generator)
        # pandas dataframe
        df_from_data = next(data_csv_generator)
        df_from_predictions = next(predictions_csv_generator)
        prep = prep_from_outputs.Prep(df_from_attributes, df_from_predictions, df_from_data)
        prep.to_neo4j()

# You can uncomment this to run the process of importing data from output files into Neo4j based on our ontology
output_to_neo4j()
