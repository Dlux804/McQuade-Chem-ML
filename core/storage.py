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
import xml.etree.cElementTree as ET
from lxml import etree


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
    for k, v in tqdm(d.items(), desc="Export to JSON", position=0):

        # grab pandas related objects for export to csv
        if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
            if k != "smiles_series":
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

    def __root_to_xml__(root, file):
        tree = ET.ElementTree(root)
        tree.write(file)
        parser = etree.XMLParser(resolve_entities=False, strip_cdata=False)
        document = etree.parse(file, parser)
        document.write(file, pretty_print=True, encoding='utf-8')

    def __get_compound_set__(smiles):
        if smiles in self.test_molecules:
            return 'TE'
        if smiles in self.train_molecules:
            return 'TR'
        else:
            return 'V'

    def __compound_to_dir__(compound):
        compounds_dir = mkd(f'{compound["id"]}')
        with open('smiles', 'w') as f:
            f.write(compound["smiles"])
        with open('molfile', 'w') as f:
            mol = MolFromSmiles(compound["smiles"])
            f.write(MolToMolBlock(mol))
        os.chdir(compounds_dir)
        comp = ET.SubElement(root, "Compound")
        ET.SubElement(comp, "Id").text = compound["id"]
        ET.SubElement(comp, "Cargos").text = "smiles molfile"

    # Get current directory, change in qdb directory
    cur_dir = mkd(f'{self.run_name}_qdb')

    # Gather all data (without validation data), testing data, and training data
    data_df = self.data
    data_df['id'] = data_df['smiles'].apply(__get_compound_set__)
    data_df = data_df.sort_values(by=['id'])
    data_df = data_df.reset_index(drop=True)
    data_df['id'] = data_df['id'] + '_' + data_df.index.astype(str)

    # Create archive.xml
    root = ET.Element("Archive", xmlns="http://www.qsardb.org/QDB")
    ET.SubElement(root, "Name").text = "No Names available yet"
    ET.SubElement(root, "Description").text = "Give our workflow a fancy description"
    __root_to_xml__(root, "archive.xml")

    # Work on compounds directory
    main_dir = mkd('compounds')
    root = ET.Element("CompoundRegistry", xmlns="http://www.qsardb.org/QDB")
    data_df.apply(__compound_to_dir__, axis=1)
    __root_to_xml__(root, "compounds.xml")
    os.chdir(main_dir)

    # Work on descriptors directory
    main_dir = mkd('descriptors')
    non_descriptors_columns = ['smiles', 'id', 'in_set', self.target_name]
    root = ET.Element("DescriptorRegistry", xmlns="http://www.qsardb.org/QDB")
    for col in data_df.columns:
        if col not in non_descriptors_columns:
            descriptors_dir = mkd(col)
            descriptor_df = data_df[['id', col]]
            descriptor_df.to_csv('values', sep='\t', index=False)
            os.chdir(descriptors_dir)
            desc = ET.SubElement(root, "Descriptor")
            ET.SubElement(desc, "Id").text = col
            ET.SubElement(desc, "Cargos").text = "values"
            ET.SubElement(desc, "Name").text = "No Names available yet"
    __root_to_xml__(root, "descriptors.xml")
    os.chdir(main_dir)

    # Work on properties directory
    main_dir = mkd('properties')
    prop_dir = mkd(f'{self.target_name}')
    properties_df = data_df[['id', self.target_name]]
    properties_df.to_csv('values', sep='\t', index=False)
    os.chdir(prop_dir)
    root = ET.Element("PropertyRegistry", xmlns="http://www.qsardb.org/QDB")
    prop = ET.SubElement(root, "Property")
    ET.SubElement(prop, "Id").text = self.target_name
    ET.SubElement(prop, "Name").text = "No Names available yet"
    ET.SubElement(prop, "Cargos").text = "values"
    __root_to_xml__(root, "properties.xml")
    os.chdir(main_dir)

    # Work on predictions
    main_dir = mkd('predictions')
    pred_dir = mkd(f'{self.algorithm}_test')
    predictions_df = data_df.merge(self.predictions, on='smiles', how='inner')[['id', 'pred_avg']]
    predictions_df.to_csv('values', sep='\t', index=False)
    os.chdir(pred_dir)
    root = ET.Element("PredictionRegistry", xmlns="http://www.qsardb.org/QDB")
    pred = ET.SubElement(root, "Prediction")
    ET.SubElement(pred, "Id").text = f"{self.algorithm}_Test"
    ET.SubElement(pred, "Name").text = "Testing Set"
    ET.SubElement(pred, "Cargos").text = "values"
    ET.SubElement(pred, "ModelId").text = self.algorithm
    ET.SubElement(pred, "Type").text = "testing"
    ET.SubElement(pred, "Application").text = "Python 3"
    __root_to_xml__(root, "predictions.xml")
    os.chdir(main_dir)

    # Work on models
    main_dir = mkd('models')
    model_dir = mkd(self.algorithm)
    with open('bibtex', 'w') as f:  # Holding for writing bibtex later on
        pass
    with open('pmml', 'w') as f:  # Holding to better understand ppml
        pass
    os.chdir(model_dir)
    root = ET.Element("ModelRegistry", xmlns="http://www.qsardb.org/QDB")
    model = ET.SubElement(root, "Model")
    ET.SubElement(model, "Id").text = self.algorithm
    ET.SubElement(model, "Name").text = "No Names available yet"
    ET.SubElement(model, "Cargos").text = "pmml bibtex"
    ET.SubElement(model, "PropertyId").text = self.target_name
    __root_to_xml__(root, "models.xml")
    os.chdir(main_dir)

    # Switch back to original directory
    os.chdir(cur_dir)
