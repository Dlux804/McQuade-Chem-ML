import pandas as pd
import numpy as np
import os, shutil
import csv
from contextlib import contextmanager

@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = os.getcwd()
    print('Previous PATH:', prevdir)
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        print('Switching back to previous PATH:', prevdir)
        os.chdir(prevdir)

try:
   os.mkdir("./merged_files")
except OSError as e:
   print("Directory exists")


def csv_names():
    data = 'ml_results3.csv'
    algorithm = 'rf'
    df = pd.read_csv(data)
    df = df[df.algorithm == algorithm]
    df = df[df.dataset != 'water-energy.csv']
    df_reset = df.reset_index(drop=True)
    # algo_list = df_reset['algorithm'].tolist()
    data_list = df_reset['dataset'].tolist()
    featmeth_list = df_reset['feat_meth'].tolist()
    name_lst = []
    for data, featmeth in zip(data_list, featmeth_list):
        feats = '' + str(featmeth)
        name = data[:-4] + '_' + algorithm + '_' + feats
        name_lst.append(name)
    print('Name List:', name_lst)
    return name_lst



def make_file_list(srcDir):
#     # For every folder in the directory
    name_list = csv_names()
    for name in name_list:
        for root, dirs, files in os.walk(srcDir):
            for f in files:
                if name in f:
                    print(f)
                    if not os.path.exists(name):
                        os.mkdir(name)
                        shutil.copy(os.path.join(root, f), name)
                    else:
                        # with cd('merged_files')
                        shutil.copy(os.path.join(root, f), name)
                else:
                    pass

# with cd('merged_files'):
# make_file_list('C:/Users/quang/McQuade-Chem-ML/rerun_2')
#
#
#
# file_list = make_file_list('C:\Users\quang\McQuade-Chem-ML')

