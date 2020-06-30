import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import subprocess
from core.misc import cd
from pprint import pprint
from time import sleep
import shutil
import zipfile


class NumpyEncoder(json.JSONEncoder):
    """
    Modifies JSONEncoder to convert numpy arrays to lists first.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def __clean_up_param_grid_item__(item):
    # This function will pull attributes of out Integer and Categorical classes generated from skopt
    d = dict(vars(item))  # Get attributes in sub-class
    labels_to_pop = []
    for label, item in d.items():
        try:
            dict(vars(item))  # Test if sub-sub attribute is also a class
            labels_to_pop.append(label)  # If so remove it
        except TypeError:
            pass
    for label in labels_to_pop:
        d.pop(label)
    return d


def export_json(self):
    """
    export entire model instance as a json file.  First converts all attributes to dictionary format.
    Some attributes will have unsupported data types for JSON export.  Notably, numpy arrays, Pandas
    Series and DataFrames and any other Python object, like graphs, molecule objects, etc.  Pandas
    objects are exported to their own JSON via pandas API.  All other unsupported objects are dropped.
    :param self:
    :return: writes json file
    """
    print("Model attributes are being exported to JSON format...")
    NoneType = type(None)  # weird but necessary decalartion for isinstance() to work on None
    d = dict(vars(self))  # create dict of model attributes
    # pprint(d)
    objs = []
    dfs = []
    for k, v in tqdm(d.items(), desc="Export to JSON", position=0):
        # print('Attribute {} is of type {}'.format(k, type(v)))

        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            objs.append(k)
            dfs.append(k)
            v.to_csv(self.run_name + '_' + k + '.csv')
            # getattr(self, k).to_json(path_or_buf=self.run_name + '_' + k + '.json')

        if not isinstance(v, (int, float, dict, tuple, list, np.ndarray, bool, str, NoneType)):
            objs.append(k)

        if k == 'param_grid':  # Param grid does not behave properly,
            new_param_grid_dict = {}
            for label, item in d[k].items():
                new_param_grid_dict[label] = __clean_up_param_grid_item__(item)  # Cleanup each item in param_gird dict
            d[k] = new_param_grid_dict
        if k == 'fit_params' and not isinstance(v, NoneType):
            # grab epochs and store, then remove fit_params
            # TODO update this to get early stopping criteria
            epochs = v['epochs']
            # update will change size, so delete to preserve size.
            d.update({'max_epochs': epochs})
            del d[k]



    objs = list(set(objs))
    print("Unsupported JSON Export Attributes:", objs)
    print("The following pandas attributes were exported to individual CSVs: ", dfs)
    for k in objs:
        del d[k]

    json_name = self.run_name + '_attributes' + '.json'
    with open(json_name, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder)


def pickle_model(self):
    with open(self.run_name + '.pkl', 'wb') as f:
        pickle.dump(self, f)


def unpickle_model(file_location):
    with open(file_location, 'rb') as f:
        return pickle.load(f)


def org_files(self, zip_only=False):
    try:
        os.mkdir(self.run_name)
    except OSError as e:
        pass

    # put output files into new folder
    filesp = ''.join(['move ./', self.run_name, '*.* ', self.run_name, '/'])  # move for Windows system
    # filesp = ''.join(['mv ./', self.run_name, '*.* ', self.run_name, '/'])  # mv for Linux system
    subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

    # zip the output folder to save space
    path = os.getcwd()
    base = self.run_name
    zipdir = path+'\\'+base
    print('Zipping output to {}'.format(base+'.zip'))
    sleep(3)  # need to give time for files to move before zip.
    shutil.make_archive(base, 'zip', zipdir)

    if zip_only:  # remove unzipped folder to avoid duplicates
        rmdir = 'rmdir ' + self.run_name + ' /s/q'
        subprocess.Popen(rmdir, shell=True, stdout=subprocess.PIPE)  # run bash command
