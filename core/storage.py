import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import subprocess
from core.misc import cd
import platform

from time import sleep
import shutil
from rdkit.Chem import MolFromSmiles, MolToMolBlock


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
    np_arrays = []
    for k, v in tqdm(d.items(), desc="Export to JSON", position=0):

        # grab pandas related objects for export to csv
        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            objs.append(k)
            dfs.append(k)
            v.to_csv(self.run_name + '_' + k + '.csv')

        # grab non-Java compatible attributes
        if not isinstance(v, (int, float, dict, tuple, list, np.ndarray, bool, str, NoneType)):
            objs.append(k)

        if isinstance(v, np.ndarray):
            np_arrays.append(k)

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

    # reduce list of exceptions to unique entries
    objs = list(set(objs))
    print("Unsupported JSON Export Attributes:", objs)
    print("The following pandas attributes were exported to individual CSVs: ", dfs)
    print("The following numpy attributes were not exported: ", np_arrays)
    for k in objs:
        del d[k]  # remove prior to export
    for k in np_arrays:
        del d[k]  # remove numpy arrays

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

    # make directory for run outputs
    try:
        os.mkdir(self.run_name)
    except OSError as e:
        pass

    # put output files into new folder
    system = platform.system()
    if system == "Windows":
        filesp = ''.join(['move ./', self.run_name, '*.* ', self.run_name, '/'])  # move for Windows system
    else:
        filesp = ''.join(['mv ./', self.run_name, '*.* ', self.run_name, '/'])  # mv for Linux system
    subprocess.Popen(filesp, shell=True, stdout=subprocess.PIPE)  # run bash command

    # zip the output folder to save space
    path = os.getcwd()
    base = self.run_name
    zipdir = path+'/'+base
    print('Zipping output to {}'.format(base+'.zip'))
    sleep(3)  # need to give time for files to move before zip.
    shutil.make_archive(base, 'zip', zipdir)

    if zip_only:  # remove unzipped folder to avoid duplicates
        if system == "Windows":
            rmdir = 'rmdir ' + self.run_name + ' /s/q'
            subprocess.Popen(rmdir, shell=True, stdout=subprocess.PIPE)  # run bash command
        else:
            subprocess.Popen(self.run_name, shell=True, stdout=subprocess.PIPE)  # run bash command

def QsarDB_export(self):
    # Define function for less code to change between directories
    def mkd(new_dir):
        cur_dir = os.getcwd()
        try:
            os.mkdir(new_dir)
        except OSError:
            pass
        os.chdir(new_dir)
        return cur_dir

    cur_dir = mkd(f'{self.run_name}_qdb')

    # Gather all data (without validation data), testing data, and training data
    testing_data = []
    training_data = []
    data_dicts = []
    training_counter = 0
    testing_counter = 0
    for compound in self.data.to_dict('records'):
        smiles = compound['smiles']
        if smiles in self.test_molecules:
            compound['id'] = f'V{testing_counter}'
            testing_counter = testing_counter + 1
            testing_data.append(compound)
            data_dicts.append(compound)
        if smiles in self.train_molecules:
            compound['id'] = f'T{training_counter}'
            training_counter = training_counter + 1
            training_data.append(compound)
            data_dicts.append(compound)
    data_df = pd.DataFrame(data_dicts)

    # Work on compounds directory
    main_dir = mkd('compounds')
    for compound in data_dicts:
        compounds_dir = mkd(f'{compound["id"]}')
        with open('smiles', 'w') as f:
            f.write(compound["smiles"])
        with open('molfile', 'w') as f:
            mol = MolFromSmiles(compound["smiles"])
            f.write(MolToMolBlock(mol))
        os.chdir(compounds_dir)
    os.chdir(main_dir)

    # Work on descriptors directory
    main_dir = mkd('descriptors')
    non_descriptors_columns = ['smiles', 'id', self.target_name]
    for col in data_df.columns:
        if col not in non_descriptors_columns:
            descriptors_dir = mkd(col)
            descriptor_df = data_df[['id', col]]
            descriptor_df.to_csv('values', sep='\t', index=False)
            os.chdir(descriptors_dir)
    os.chdir(main_dir)

    # Work on properties directory
    main_dir = mkd('properties')
    mkd(f'{self.target_name}')
    properties_df = data_df[['id', self.target_name]]
    properties_df.to_csv('values', sep='\t', index=False)
    os.chdir(main_dir)

    # Work on predictions
