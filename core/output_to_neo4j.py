"""
Objective: Create Neo4j graphs from files inside of "output"
"""

import os
import pathlib
import py2neo
from zipfile import ZipFile
from core import misc


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
                        json_file = "_attributes.json"
                        csv_file = "_predicted.csv"
                        for file in file_list:
                            if json_file in file:
                                print(file)
access_ouput()