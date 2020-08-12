import zipfile
import shutil
import os

import xml.etree.cElementTree as ET
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromMolFile, MolToSmiles, MolFromSmiles
from sklearn.metrics import mean_squared_error, r2_score

from core.neo4j import ModelOrOutputToNeo4j


class QsarDBImport:

    def __init__(self, directory, zipped=False, cleanup_unzipped_dir=True):

        def __combine_dfs__(df1, df2, how='inner'):
            if df1['Compound Id'].dtype != 'object':
                df1['Compound Id'] = df1['Compound Id'].astype(str)
            if df2['Compound Id'].dtype != 'object':
                df2['Compound Id'] = df2['Compound Id'].astype(str)
            return df1.merge(df2, on='Compound Id', how=how)

        raw_dir = None
        if zipped:

            unzipped_directory_file_name = directory.split('.')
            extension = unzipped_directory_file_name.pop(-1)

            if extension != 'zip':
                raise Exception("Zipped folders should have a .zip extension")

            unzipped_directory_file_name = ".".join(unzipped_directory_file_name)

            with zipfile.ZipFile(directory, 'r') as zip_ref:
                zip_ref.extractall(unzipped_directory_file_name)

            directory = unzipped_directory_file_name

            qdb_dir = directory.split('/')[-1]
            if os.path.exists(directory + '/' + qdb_dir):
                raw_dir = directory
                directory = directory + '/' + qdb_dir

        archive_xml = directory + '/archive.xml'
        archive_xml = ET.parse(archive_xml).getroot()
        for parent in archive_xml:
            tag = parent.tag.split('}')[1]
            if tag == 'Name':
                all_models_dataset = parent.text

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
        compound_data = pd.DataFrame(compound_dicts)
        compound_data = compound_data.rename(columns={'Id': 'Compound Id'})

        # gather information in cargos
        compounds_cargos_dicts = []
        for compound_dir in [f.name for f in os.scandir(root_compounds_dir) if f.is_dir()]:
            compound_cargos = {'Compound Id': compound_dir}
            compound_dir = root_compounds_dir + '/' + compound_dir
            for cargo in os.listdir(compound_dir):
                cargo_label = cargo
                cargo = compound_dir + '/' + cargo
                with open(cargo, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 1:
                        compound_cargos[cargo_label] = lines[0]
                    else:
                        mol = MolFromMolFile(cargo)
                        if mol is not None:
                            compound_cargos['rdkit-smiles'] = MolToSmiles(mol)
            compounds_cargos_dicts.append(compound_cargos)
        if compounds_cargos_dicts:
            cargos_df = pd.DataFrame(compounds_cargos_dicts)
            compound_data = __combine_dfs__(compound_data, cargos_df)

        # Gather information in descriptors columns
        desc_dir = directory + '/' + 'descriptors'
        desc_xml = desc_dir + '/descriptors.xml'
        desc_xml = ET.parse(desc_xml).getroot()

        descriptors = []
        temp_descriptors_dicts = {}
        for desc in desc_xml:
            Id = None
            Name = None
            for prop in desc:
                label = prop.tag.split('}')[1]
                value = prop.text
                if label == 'Id':
                    Id = value
                if label == 'Name':
                    Name = value
                    descriptors.append(Name)
            temp_descriptors_dicts[Id] = Name

        feats = []
        for desc in [f.name for f in os.scandir(desc_dir) if f.is_dir()]:
            desc_id = desc
            desc_csv = desc_dir + f'/{desc}/values'
            desc_csv = pd.read_csv(desc_csv, sep='\t')

            current_col = list(desc_csv.columns)
            current_col.remove('Compound Id')
            current_col = current_col[0]
            feats.append(temp_descriptors_dicts[desc_id])

            desc_csv = desc_csv.rename(columns={current_col: temp_descriptors_dicts[desc_id]})
            compound_data = __combine_dfs__(compound_data, desc_csv)

        # Clean up compound_data for other dataframes
        if 'rdkit-smiles' in list(compound_data.columns):
            compound_data = compound_data.rename(columns={'rdkit-smiles': 'smiles'})

        elif 'daylight-smiles' in list(compound_data.columns):
            def __daylight_to_rdkit__(smiles):
                return MolToSmiles(MolFromSmiles(smiles))

            compound_data['daylight-smiles'] = compound_data['daylight-smiles'].apply(__daylight_to_rdkit__)
            compound_data = compound_data.rename(columns={'daylight-smiles': 'smiles'})

        else:
            # TODO convert cas/inchi/name to smiles if needed
            compound_data = compound_data['Compound Id']

        all_data = compound_data
        # all_data.to_csv('dev.csv')
        compound_data = compound_data[['Compound Id', 'smiles']]

        # Gather Model(s) structures
        models = []
        root_models_dir = directory + '/' + 'models'
        models_xml = root_models_dir + '/' + 'models.xml'
        models_xml = ET.parse(models_xml).getroot()
        for model in models_xml:
            model_props = {}
            for prop in model:
                label = prop.tag.split('}')[1]
                value = prop.text
                if label == 'Id':
                    model_props['Id'] = value
                if label == 'PropertyId':
                    model_props['PropertyId'] = value.split('\t')[0]
                if label == 'Name':
                    model_props['Name'] = value
            models.append(model_props)

        for model in models:
            pmml = root_models_dir + f'/{model["Id"]}/pmml'
            pmml = ET.parse(pmml).getroot()
            model['algorithm'] = pmml[1].tag.split('}')[1]
            model['task_type'] = model['algorithm'].lower().split('model')[0]
            model['dataset'] = all_models_dataset

        # Gather raw data from models
        properties_dir = directory + '/' + 'properties'
        for model in models:
            prop = model['PropertyId']

            properties_xml = properties_dir + '/properties.xml'
            properties_xml = ET.parse(properties_xml).getroot()
            id_found = False
            for parent in properties_xml:
                for child in parent:
                    label = child.tag.split('}')[1]
                    value = child.text
                    if label == 'Id' and value == prop:
                        id_found = True
                    if id_found and label == 'Name':
                        target_name = value
                        model['target_name'] = value

            raw_data = None
            property_csv = properties_dir + f'/{prop}/values'
            property_csv = pd.read_csv(property_csv, sep='\t')
            current_property_column = list(property_csv.columns)[1]
            property_csv = property_csv.rename({current_property_column: target_name})
            if raw_data is None:
                raw_data = __combine_dfs__(compound_data, property_csv)
            else:
                raw_data = __combine_dfs__(raw_data, property_csv)
            all_data = __combine_dfs__(all_data, property_csv, how='outer')
            model['raw_data'] = raw_data
            model['n_total'] = len(raw_data)

        # Gather test/train/val data
        predictions_dir = directory + '/' + 'predictions'
        predictions_xml = predictions_dir + '/predictions.xml'
        predictions_xml = ET.parse(predictions_xml).getroot()
        for model in models:
            model['n'] = {'training': 0, 'testing': 0, 'validation': 0}
            model['r2'] = {'training': None, 'testing': None, 'validation': None}
            model['mse'] = {'training': None, 'testing': None, 'validation': None}
            model['rmse'] = {'training': None, 'testing': None, 'validation': None}
            model_sets = {'training': None, 'testing': None, 'validation': None}
            for ml_set in predictions_xml:
                ml_set_name = None
                correct_model = False
                for prop in ml_set:
                    label = prop.tag.split('}')[1]
                    value = prop.text
                    if label == 'Id':
                        ml_set_name = value
                    if label == 'ModelId':
                        if value == model['Id']:
                            correct_model = True
                    if label == 'Type' and correct_model:
                        model_sets[value] = ml_set_name

            for set_type, set_name in model_sets.items():
                if set_name is None:
                    model[f'{set_type}_predicted'] = None
                    model[f'{set_type}_actual'] = None
                else:
                    predictions_csv = predictions_dir + f'/{set_name}/values'
                    predictions_csv = pd.read_csv(predictions_csv, sep='\t')
                    predicted_data = __combine_dfs__(compound_data, predictions_csv)
                    predicted_data_compound_ids = predicted_data[['Compound Id']]
                    actual_data = model['raw_data']
                    actual_data = __combine_dfs__(actual_data, predicted_data_compound_ids)
                    model[f'{set_type}_predicted'] = predicted_data
                    model[f'{set_type}_actual'] = actual_data

                    if 'smiles' in compound_data.columns:
                        pred_column = list(predicted_data.columns.difference(['Compound Id', 'smiles']))[0]
                        act_column = list(actual_data.columns.difference(['Compound Id', 'smiles']))[0]
                    else:
                        pred_column = list(predicted_data.columns.difference(['Compound Id']))[0]
                        act_column = list(actual_data.columns.difference(['Compound Id']))[0]

                    r2 = r2_score(actual_data[act_column], predicted_data[pred_column])
                    mse = mean_squared_error(actual_data[act_column], predicted_data[pred_column])
                    rmse = np.sqrt(mean_squared_error(actual_data[act_column], predicted_data[pred_column]))
                    model['n'][set_type] = len(predicted_data)
                    model['r2'][set_type] = r2
                    model['mse'][set_type] = mse
                    model['rmse'][set_type] = rmse

            val_molecules = list(model['validation_predicted']['smiles'])
            train_molecules = list(model['training_predicted']['smiles'])

            # Logic to seperate data in test/train/val
            def __fetch_set__(smiles):
                if smiles in val_molecules:
                    return 'test'
                elif smiles in train_molecules:
                    return 'train'
                else:
                    return 'val'

            model['all_data'] = all_data
            model['all_data']['in_set'] = model['all_data']['smiles'].apply(__fetch_set__)
            model['feats'] = feats

        if cleanup_unzipped_dir and zipped:
            if raw_dir is not None:
                shutil.rmtree(raw_dir, ignore_errors=True)
            else:
                shutil.rmtree(directory, ignore_errors=True)

        self.all_data = all_data
        self.models = models

    def to_neo4j(self):
        ModelOrOutputToNeo4j(qsar_obj=self.models)
