from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs, clean_up_checker_files
from Neo4j.US_patents.file_to_neo4j import file_to_neo4j
import pandas as pd
import os

US_patents_directory = 'C:/Users/User/Desktop/5104873'

'''
The above line should be the only line you should only have to edit to make this script work. For the US patents 
directory, make sure you only have the folders 'grants' and 'applications' saved in the directory. The script is
only designed to work for a directory containing only those unzipped folders. 

Then afterward, start up a new neo4j graph with the password as 'password' and the script will do the rest :D.
*Note this script is dirty and inefficient, efficiency is being worked out. This is only for prototyping. 
'''

US_grants_directory_to_csvs(US_patents_directory)
clean_up_checker_files(US_patents_directory)
for main_directories in os.listdir(US_patents_directory):
    main_directories = US_patents_directory + '/' + main_directories
    for directories in os.listdir(main_directories):
        if directories[-4:] == '_csv':
            directory = main_directories + '/' + directories
            i = len(os.listdir(directory))
            counter = 0
            for file in os.listdir(directory):
                print(directory + ": There are {} files left in directory".format(str(i-counter)))
                file = directory + '/' + file
                try:
                    file_to_neo4j(file)
                except pd.errors.EmptyDataError:
                    pass
                counter = counter + 1
