import os
import re
import shutil
import zipfile

import pandas as pd
import xml.etree.cElementTree as ET
from lxml import etree
from rdkit.Chem import MolFromSmiles, MolToMolBlock
from sklearn2pmml import PMMLPipeline, sklearn2pmml

from core.storage.misc import compress_fingerprint


def QsarDB_export(self, zip_output=False):
    """
    :param zip_output: Weather to zip output folder
    :param self: Model Object
    :return:

    Purpose of this is to export our models in a format that complies with the QsarDB requirements.
    This script is meant to be run on a model that has been fully featurized and tested.
    """

    print("Exporting data into QsarDB format")

    # Define function for less code to change between directories
    def __mkd__(new_dir):
        cur_dir = os.getcwd()
        try:
            os.mkdir(new_dir)
        except OSError:
            pass
        os.chdir(new_dir)
        return cur_dir

    # Take a root, and spit that to a pretty xml file
    def __root_to_xml__(root, file):
        tree = ET.ElementTree(root)
        tree.write(file)
        parser = etree.XMLParser(resolve_entities=False, strip_cdata=False)
        document = etree.parse(file, parser)
        document.write(file, pretty_print=True, encoding='utf-8')

    # Fetch which set each molecule is in
    def __get_compound_set__(smiles):
        if smiles in self.test_molecules:
            return 'TE'
        if smiles in self.train_molecules:
            return 'TR'
        else:
            return 'V'

    # Crate compound directories with mod-files and smiles files
    def __compound_to_dir__(compound):
        compounds_dir = __mkd__(f'{compound["id"]}')
        with open('smiles', 'w') as f:
            f.write(compound["smiles"])
        with open('molfile', 'w') as f:
            mol = MolFromSmiles(compound["smiles"])
            f.write(MolToMolBlock(mol))
        os.chdir(compounds_dir)
        comp = ET.SubElement(root, "Compound")
        ET.SubElement(comp, "Id").text = compound["id"]
        ET.SubElement(comp, "Cargos").text = "smiles molfile"

    def __model_to_pmml__():

        pipeline = PMMLPipeline([
            ("regressor", self.regressor)
        ])
        sklearn2pmml(pipeline, "pmml", with_repr=True)

        print('creating pmml')
        # Read in the file
        with open('pmml', 'r') as file:
            filedata = file.read()

        print('finding matches')
        # Replace x[1-...] with actual column names
        m = re.findall('x\-?\d+', filedata)
        matches = []
        print('sorting matches')
        for match in m:
            if match in matches:
                break
            matches.append(match)
        feature_cols = list(data_df.columns.difference(["in_set", "smiles", "id", self.target_name]))
        matched_dict = dict(zip(matches, feature_cols))
        print('replacing')
        for match, feat in matched_dict.items():
            filedata = filedata.replace(match, f'{feat}')

        # Replace y with target name
        filedata = filedata.replace('y', f'{self.target_name}')

        print('rewrite to file')
        # Write the file out again
        with open('pmml', 'w') as file:
            file.write(filedata)

    # Get current directory, change in qdb directory
    cur_dir = __mkd__(f'{self.run_name}_qdb')
    non_descriptors_columns = ['smiles', 'id', 'in_set', self.target_name]

    # Gather all data (without validation data), testing data, and training data
    data_df = compress_fingerprint(self.data)
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
    main_dir = __mkd__('compounds')
    root = ET.Element("CompoundRegistry", xmlns="http://www.qsardb.org/QDB")
    data_df.apply(__compound_to_dir__, axis=1)
    __root_to_xml__(root, "compounds.xml")
    os.chdir(main_dir)

    # Work on descriptors directory
    main_dir = __mkd__('descriptors')
    root = ET.Element("DescriptorRegistry", xmlns="http://www.qsardb.org/QDB")
    for col in data_df.columns:
        if col not in non_descriptors_columns:
            descriptors_dir = __mkd__(col)
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
    main_dir = __mkd__('properties')
    prop_dir = __mkd__(f'{self.target_name}')
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
    main_dir = __mkd__('predictions')
    pred_dir = __mkd__(f'{self.algorithm}_test')
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
    __mkd__('models')

    # Make models.xml
    root = ET.Element("ModelRegistry", xmlns="http://www.qsardb.org/QDB")
    model = ET.SubElement(root, "Model")
    ET.SubElement(model, "Id").text = self.algorithm
    ET.SubElement(model, "Name").text = "No Names available yet"
    ET.SubElement(model, "Cargos").text = "pmml bibtex"
    ET.SubElement(model, "PropertyId").text = self.target_name
    __root_to_xml__(root, "models.xml")
    __mkd__(self.algorithm)
    __model_to_pmml__()

    # Switch back to original directory
    os.chdir(cur_dir)

    if zip_output:
        shutil.make_archive((os.getcwd() + '/ ' + f'{self.run_name}_qdb'), 'zip',
                            os.getcwd(), f'{self.run_name}_qdb')
        shutil.rmtree(f'{self.run_name}_qdb', ignore_errors=True)


def QsarDB_import(directory, zipped=False, cleanup_unzipped_dir=True):
    def __combine_dfs__(df1, df2):
        df1['Compound Id'] = df1['Compound Id'].astype(int)
        df2['Compound Id'] = df2['Compound Id'].astype(int)
        return df1.merge(df2, on='Compound Id', how='outer')

    if zipped:

        unzipped_directory_file_name = directory.split('.')
        extension = unzipped_directory_file_name.pop(-1)

        if extension != 'zip':
            raise Exception("Zipped folders should have a .zip extension")

        unzipped_directory_file_name = ".".join(unzipped_directory_file_name)

        with zipfile.ZipFile(directory, 'r') as zip_ref:
            zip_ref.extractall(unzipped_directory_file_name)

        directory = unzipped_directory_file_name

    """Work on compounds dir"""
    root_compounds_dir = directory + '/' + 'compounds'

    # get compound details from compounds.xml
    compounds_xml = root_compounds_dir + '/' + 'compounds.xml'
    compounds_xml_root = ET.parse(compounds_xml).getroot()
    compound_dicts = []
    for compound in compounds_xml_root:
        compound_props = {}
        for prop in compound:
            label = prop.tag.split('}')[1]
            value = prop.text
            if label != 'Cargos':
                compound_props[label] = value
        compound_dicts.append(compound_props)

    # create dataframe to add data too, and append columns in cargos
    data = pd.DataFrame(compound_dicts)
    data = data.rename(columns={'Id': 'Compound Id'})

    # gather information in cargos
    compounds_cargos_dicts = []
    for compound_dir in [f.name for f in os.scandir(root_compounds_dir) if f.is_dir()]:
        compound_cargos = {'Compound Id': compound_dir}
        compound_dir = root_compounds_dir + '/' + compound_dir
        for cargo in os.listdir(compound_dir):
            cargo_label = cargo
            cargo = compound_dir + '/' + cargo
            with open(cargo, 'r') as f:
                compound_cargos[cargo_label] = f.read()
        compounds_cargos_dicts.append(compound_cargos)
    if compounds_cargos_dicts:
        cargos_df = pd.DataFrame(compounds_cargos_dicts)
        data = __combine_dfs__(data, cargos_df)

    """Work on descriptors, predictions, and properties dir"""

    values_directories = ['descriptors', 'predictions', 'properties']
    target_columns = []

    for value_directory in values_directories:
        root_dir = directory + '/' + value_directory

        for sub_value_dir in [f.name for f in os.scandir(root_dir) if f.is_dir()]:
            value_csv = root_dir + '/' + sub_value_dir + '/values'
            values = pd.read_csv(value_csv, sep='\t')
            data = __combine_dfs__(data, values)

            if value_directory == 'properties':
                prop_columns = list(values.columns)
                prop_columns.remove('Compound Id')
                target_columns.append(prop_columns[0])

    """Extract pmml from models directory"""

    model_directories = directory + '/' + 'models'
    for sub_value_dir in [f.name for f in os.scandir(model_directories) if f.is_dir()]:
        model_name = sub_value_dir
        pmml_file = model_directories + '/' + sub_value_dir + '/pmml'
        # TODO import models from the pmml file

    if len(target_columns) == 1:
        target_columns = target_columns[0]
    elif len(target_columns) == 0:
        raise Exception("No target column was found...")

    if cleanup_unzipped_dir and zipped:
        shutil.rmtree(directory, ignore_errors=True)

    return data, target_columns

