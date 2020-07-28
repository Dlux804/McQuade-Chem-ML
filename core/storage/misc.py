import json
import pickle
import os
import re
import subprocess
import platform
import shutil
from time import sleep
from contextlib import contextmanager

import pandas as pd
import numpy as np
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """
    Modifies JSONEncoder to convert numpy arrays to lists first.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def foo(self, string):
    print('\n', self.algorithm)
    print(string)
    return string


@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = os.getcwd()
    # print('Previous PATH:', prevdir)
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        # print('Switching back to previous PATH:', prevdir)
        os.chdir(prevdir)


def __clean_up_params__(params):
    if isinstance(params, tuple):  # Check is a tuple
        params = list(params)
        new_params = []
        for param in params:
            try:
                new_params.append(dict(vars(param)))  # Make sure item in tuple is a class
            except TypeError:
                new_params.append(param)
        return new_params
    else:
        return params


def __clean_up_param_grid_item__(item):
    # This function will pull attributes of out Integer and Categorical classes generated from skopt

    if isinstance(item, tuple):  # Test if tuple of classes
        return __clean_up_params__(item)

    try:  # Test if item is actually a class
        d = dict(vars(item))  # Get attributes in sub-class
    except TypeError:
        return item

    new_d = {}
    for label, item in d.items():
        try:
            dict(vars(item))  # Test if sub-sub attribute is also a class
        except TypeError:
            new_d[label] = item
        if isinstance(item, tuple):  # Test if tuple of classes
            new_d[label] = __clean_up_params__(item)
    return new_d


def store(self):
    """
    export entire model instance as a json file.  First converts all attributes to dictionary format.
    Some attributes will have unsupported data types for JSON export.  Notably, numpy arrays, Pandas
    Series and DataFrames and any other Python object, like graphs, molecule objects, etc.  Pandas
    objects are exported to their own JSON via pandas API.  All other unsupported objects are dropped.
    :param self:
    :return: writes json file
    """

    print("Model attributes are being exported to JSON format...")
    NoneType = type(None)  # weird but necessary declaration for isinstance() to work on None
    d = dict(vars(self))  # create dict of model attributes
    objs = []  # empty lists to capture excepted attributes
    dfs = []
    for k, v in tqdm(d.items(), desc="Export to JSON", position=0):

        # grab pandas related objects for export to csv
        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            if k == "data":  # Rename target column to a more general one
                if self.dataset not in self.multi_label_classification_datasets:
                    v = v.rename(columns={self.target_name: "target"})
                v.to_csv(self.run_name + '_' + k + '.csv')

            elif k != "smiles_series":
                objs.append(k)
                dfs.append(k)
                v.to_csv(self.run_name + '_' + k + '.csv')
            else:
                objs.append(k)

        # grab non-Java compatible attributes
        if not isinstance(v, (int, float, dict, tuple, list, np.ndarray, bool, str, NoneType)):
            objs.append(k)

        if isinstance(v, np.ndarray):  # Do not store numpy arrays in attributes.json
            objs.append(k)

        if k == 'param_grid' or k == 'params':  # Param grid does not behave properly,
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

    # reduce list of exceptions to unique entries
    objs = list(set(objs))
    print("Attributes were not exported to JSON:", objs)
    print("The following pandas attributes were exported to individual CSVs: ", dfs)
    for k in objs:
        del d[k]  # remove prior to export

    json_name = self.run_name + '_attributes' + '.json'
    with open(json_name, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder)


def pickle_model(self):
    """
    Stores model class instance as a .pkl file.  File sizes may be quite large (>100MB)
    :param self:
    :return:
    """

    with open(self.run_name + '.pkl', 'wb') as f:
        pickle.dump(self, f)


def unpickle_model(file_location):
    """
    Loads model from previously saved .pkl file.
    :param file_location:
    :return:
    """

    with open(file_location, 'rb') as f:
        return pickle.load(f)


def org_files(self, zip_only=False):
    """
    Organize output files of model into an individual folder. Creates zip of folder.
    If zip_only == True then non-zipped folder will be deleted.
    :param self:
    :param zip_only: Boolean.  Deletes unzipped folder from directory.
    :return:
    """
    system = platform.system()  # Find what system user is on

    # make directory for run outputs
    try:
        os.mkdir(self.run_name)
    except OSError as e:
        pass

    # put output files into new folder
    if system == "Windows":
        filesp = ''.join(['move ./', self.run_name, '*.* ', self.run_name, '/'])  # move for Windows system
    else:
        filesp = ''.join(['mv ./', self.run_name, '*.* ', self.run_name, '/'])  # mv for Linux system
    subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

    # zip the output folder to save space
    path = os.getcwd()
    base = self.run_name
    zipdir = path + '/' + base
    print('Zipping output to {}'.format(base + '.zip'))
    sleep(3)  # need to give time for files to move before zip.
    shutil.make_archive(base, 'zip', zipdir)

    if zip_only:  # remove unzipped folder to avoid duplicates
        if system == "Windows":
            rmdir = 'rmdir ' + self.run_name + ' /s/q'
            subprocess.Popen(rmdir, shell=True, stdout=subprocess.PIPE)  # run bash command
        else:  # Directory is not properly deleted on Linux with above code
            rmdir = os.getcwd() + '/' + self.run_name
            shutil.rmtree(rmdir, ignore_errors=True)


def compress_fingerprint(df):
    # Search for fingerprint columns
    fingerprint_columns = []
    fingerprint_array_column_name = ''
    for column in df.columns:
        m = re.findall('-\-?\d+', column)
        if m:
            fingerprint_columns.append(column)
            fingerprint_array_column_name = f"FP$${column.split('-')[0]}"

    if fingerprint_columns:
        # Collect all fingerprint columns, and send them to a list of list (of fingerprints)
        fingerprint_column_lists = df[fingerprint_columns].values.tolist()
        # Make a new dataframe with the columns smiles and the single fingerprint column
        fp_df = pd.DataFrame({'smiles': list(df['smiles']),
                              fingerprint_array_column_name: fingerprint_column_lists}).astype(str)
        # Get the original dataframe, minus the fingerprint columns
        df = df[df.columns.difference(fingerprint_columns)]

        # Join the new dataframes together
        df = df.merge(fp_df, on='smiles')
    return df


def decompress_fingerprint(df):

    # Search for fingerprint column
    columns = df.columns
    fingerprint_column = None
    for column in columns:
        matches = re.findall('FP\$\$', column)
        if len(matches) > 0:
            fingerprint_column = column

    # If not found, return df as is
    if fingerprint_column is None:
        return df

    # Convert column astype(str) to columns of numbers
    fingerprints = df[fingerprint_column]
    fingerprints = fingerprints.str[1:-1]  # Remove string brackets
    fingerprints = fingerprints.str.split(', ', expand=True)
    fingerprints = fingerprints.astype(float)  # Note columns are converted to float, even if it was an int.
    fingerprints = fingerprints.dropna()

    # Rename columns to match original fingerprint column
    fingerprints_columns = []
    fingerprint_type = fingerprint_column.split('FP$$')[1]
    for column in list(fingerprints.columns):
        fingerprints_columns.append(f'{fingerprint_type}-{column}')
    fingerprints.columns = fingerprints_columns

    # Join together rest of dataframe with generated fingerprint dataframe
    df = df.drop(fingerprint_column, axis=1)
    df = pd.concat([df, fingerprints], axis=1)
    return df
