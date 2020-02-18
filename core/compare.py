import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from time import time
import pandas as pd
import shutil
import glob
import cirpy
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools

"""
Goal is to compare the output of many models in a systematic, automatic manner.
"""


#import csv files from folders
path = r'C:/Users/luxon/OneDrive/Research/McQuade/Projects/NSF/OKN/phase1/Work/ml-hte-results-20200207'  # will vary by OS, computer
allFiles = glob.glob(path + "/*/*.csv")
with open('hte-models-Master-Results.csv', 'wb') as outfile:
    for i, fname in enumerate(allFiles):
        with open(fname, 'rb') as infile:
            if i != 0:
                infile.readline()  # Throw away header on all but first file
            # Block copy rest of file from input to output without parsing
            shutil.copyfileobj(infile, outfile)
            print(fname + " has been imported.")

master = pd.read_csv('hte-models-Master-Results.csv')#, dtype={'exp': object})
# print(master)

short = master.drop(['pvaM', 'regressor', 'feature_list'], axis=1)
print(short)

def datasize(dataset):
    """Cacluate dataset size by counting rows in CSV. Accept filename (string) return int.

    Flaw is the assumption that each row is a valid data set
    entry.  It is safe for our basic cases.
    """
    with misc.cd('../dataFiles/'): # move to dataset directory
        with open(dataset) as f:
            size = sum(1 for line in f) - 1  # remove one for header
            print('The {} dataset has {} entries in it.'.format(dataset, size))
            return size


string = '../dataFiles/ESOL.csv'
datasize(string)

def moloverlap(datasets):
    """
    Identifies molecules conserved across datasets.
    Accepts list of datasets to be compared.
    Returns dataframe of conserved molecules.
    """



