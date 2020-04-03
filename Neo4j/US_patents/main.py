from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs
from Neo4j.US_patents.file_to_neo4j import file_to_neo4j
import pandas as pd
import os

US_grants_directory_to_csvs('C:/Users/User/Desktop/5104873')

head_directory = 'C:/Users/User/Desktop/5104873'
for main_directories in os.listdir(head_directory):
    main_directories = head_directory + '/' + main_directories
    for directories in os.listdir(main_directories):
        if directories[-4:] == '_csv':
            directory = main_directories + '/' + directories
            print(directory)
            for file in os.listdir(directory):
                file = directory + '/' + file
                try:
                    file_to_neo4j(file)
                except pd.errors.EmptyDataError:
                    pass