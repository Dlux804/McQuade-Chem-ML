from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs, clean_up_checker_files
from Neo4j.US_patents import csv_to_neo4j
import pandas as pd
import os


US_patents_directory = 'C:/Users/User/Desktop/5104873'

'''
For the US patents directory, 
make sure you only have the folders 'grants' and 'applications' saved in the directory. The script is
only designed to work for a directory containing only those unzipped folders. The get the folders, unzip the main folder
and unzip the folders named '2001_Sep2016_USPTOapplications_cml.7z' and '1976_Sep2016_USPTOgrants_cml.7z' using 
a program like 7-zip. Afterwards you should be left with a directory named '5104872' and the folders 'grants' and
'applications' inside of it. With all folders unzipped, and the rest deleted.

This is being worked out to make this process more user friendly, but this is the work around for now. 

Then afterward, start up a new neo4j graph with the password as 'password' and the script will do the rest :D.
*Note this script is dirty and inefficient, efficiency is being worked out. This is only really only for prototyping. 
'''

# US_grants_directory_to_csvs(US_patents_directory)  # This is here to convert the xml files to csv files
# clean_up_checker_files(US_patents_directory)  # This is here to clean up checker files if you want to recreate graph

for main_directories in os.listdir(US_patents_directory):  # Main loop, inserting the csv files
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
                    csv_to_neo4j.file_to_neo4j(file, max_nodes_in_ram=400000)
                except pd.errors.EmptyDataError:
                    pass
                counter = counter + 1
