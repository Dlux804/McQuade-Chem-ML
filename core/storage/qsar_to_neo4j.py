import zipfile
import shutil
import os

import xml.etree.cElementTree as ET
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromMolFile, MolToSmiles, MolFromSmiles
from sklearn.metrics import mean_squared_error, r2_score

from core.neo4j import ModelOrOutputToNeo4j


class QsarToNeo4j:

    def __init__(self, directory, zipped=False, cleanup_unzipped_dir=True, molecules_per_batch=5000,
                 port="bolt://localhost:7687", username="neo4j", password="password"):

        self.raw_directory = directory
        self.zipped = zipped
        self.cleanup = cleanup_unzipped_dir

        self.molecules_per_batch = molecules_per_batch
        self.port = port
        self.username = username
        self.password = password

        self.directory = self.unzip_directory()
        self.dataset = self.parse_dataset_name()
        self.compound_data = self.gather_init_compound_data()
        self.compound_data, self.feats = self.add_desc_to_compound_data()
        self.cleanup_compound_data()
        self.models = self.gather_models()
        self.gather_data_in_pmmls()
        self.gather_property_data()
        self.calculate_model_data()
        self.cleanup_dir()
        self.to_neo4j()

    @staticmethod
    def __combine_dfs__(df1, df2, how='inner'):

        if df1['Compound Id'].dtype != 'object':
            df1['Compound Id'] = df1['Compound Id'].astype(str)
        if df2['Compound Id'].dtype != 'object':
            df2['Compound Id'] = df2['Compound Id'].astype(str)
        return df1.merge(df2, on='Compound Id', how=how)

    def unzip_directory(self):
        if self.zipped:

            unzipped_directory_file_name = self.raw_directory.split('.')
            extension = unzipped_directory_file_name.pop(-1)

            if extension != 'zip':
                raise TypeError("Zipped folders should have a .zip extension")

            unzipped_directory_file_name = ".".join(unzipped_directory_file_name)

            with zipfile.ZipFile(self.raw_directory, 'r') as zip_ref:
                zip_ref.extractall(unzipped_directory_file_name)
            directory = unzipped_directory_file_name

            qdb_dir = directory.split('/')[-1]
            if os.path.exists(directory + '/' + qdb_dir):
                directory = directory + '/' + qdb_dir
            return directory
        else:
            return self.raw_directory

    def parse_dataset_name(self):
        archive_xml = self.directory + '/archive.xml'
        archive_xml = ET.parse(archive_xml).getroot()
        for parent in archive_xml:
            tag = parent.tag.split('}')[1]
            if tag == 'Name':
                return parent.text

    def gather_init_compound_data(self):
        root_compounds_dir = self.directory + '/' + 'compounds'

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
            compound_data = self.__combine_dfs__(compound_data, cargos_df)
        return compound_data

    def add_desc_to_compound_data(self):
        # Gather information in descriptors columns
        desc_dir = self.directory + '/' + 'descriptors'
        desc_xml = desc_dir + '/descriptors.xml'
        desc_xml = ET.parse(desc_xml).getroot()

        descriptors = []
        ids_to_names = {}
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
            ids_to_names[Id] = Name

        feats = []
        compound_data = self.compound_data
        for desc in [f.name for f in os.scandir(desc_dir) if f.is_dir()]:
            desc_id = desc
            desc_csv = desc_dir + f'/{desc}/values'
            desc_csv = pd.read_csv(desc_csv, sep='\t', encoding='cp1252')

            current_col = list(desc_csv.columns)
            current_col.remove('Compound Id')
            current_col = current_col[0]
            feats.append(ids_to_names[desc_id])

            desc_csv = desc_csv.rename(columns={current_col: ids_to_names[desc_id]})
            compound_data = self.__combine_dfs__(desc_csv, self.compound_data)
        return compound_data, feats

    def cleanup_compound_data(self):
        # Clean up compound_data for other dataframes
        if 'rdkit-smiles' in list(self.compound_data.columns):
            self.compound_data = self.compound_data.rename(columns={'rdkit-smiles': 'smiles'})

        elif 'daylight-smiles' in list(self.compound_data.columns):
            def __daylight_to_rdkit__(smiles):
                try:
                    return MolToSmiles(MolFromSmiles(smiles))
                except TypeError:
                    return np.nan
            self.compound_data['daylight-smiles'] = self.compound_data['daylight-smiles'].apply(__daylight_to_rdkit__)
            self.compound_data = self.compound_data.loc[self.compound_data['daylight-smiles'] != np.nan]
            self.compound_data = self.compound_data.rename(columns={'daylight-smiles': 'smiles'})

        else:
            # TODO convert cas/inchi/name to smiles if needed
            print(f'Can not parse smiles for {self.directory}')
            self.cleanup_dir()
            raise TypeError("Could not generate or parse smiles in directory. Missing rdmol file and daylight-smiles")

        self.compound_data = self.compound_data.dropna(subset=['smiles'])

    def gather_models(self):
        # Gather Model(s) structures
        models = []
        root_models_dir = self.directory + '/' + 'models'
        models_xml = root_models_dir + '/' + 'models.xml'
        models_xml = ET.parse(models_xml).getroot()
        for model in models_xml:
            for prop in model:
                label = prop.tag.split('}')[1]
                value = prop.text
                if label == 'Id':
                    model_id = value
                if label == 'PropertyId':
                    property_id = value.split('\t')[0]
                if label == 'Name':
                    name = value
            model = QsarModel(model_id=model_id, property_id=property_id, name=name, directory=self.directory)
            models.append(model)
        return models

    def gather_data_in_pmmls(self):
        root_models_dir = self.directory + '/' + 'models'
        for model in self.models:
            pmml = root_models_dir + f'/{model.model_id}/pmml'
            pmml = ET.parse(pmml).getroot()
            for i in range(len(pmml)):
                try:
                    model.algorithm = pmml[i].attrib['functionName']
                    task_type = pmml[i].tag.split('}')[1]
                    model.task_type = task_type.lower().split('model')[0]
                except KeyError:
                    pass

    def gather_property_data(self):
        properties_dir = self.directory + '/' + 'properties'
        for model in self.models:
            prop = model.property_id

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
                        model.target_name = value

            property_csv = properties_dir + f'/{prop}/values'
            property_csv = pd.read_csv(property_csv, sep='\t')
            current_property_column = list(property_csv.columns)[1]
            property_csv = property_csv.rename(columns={current_property_column: model.target_name})

            model.raw_data = self.__combine_dfs__(self.compound_data, property_csv)
            model.raw_data = model.raw_data.dropna(subset=['smiles'])
            model.n_total = len(model.raw_data)

    def calculate_model_data(self):
        predictions_dir = self.directory + '/' + 'predictions'
        predictions_xml = predictions_dir + '/predictions.xml'
        predictions_xml = ET.parse(predictions_xml).getroot()

        good_models = []
        for model in self.models:
            if len(model.raw_data) > 0:
                good_models.append(model)
        self.models = good_models

        for model in self.models:
            model.n = {'training': 0, 'testing': 0, 'validation': 0}
            model.r2 = {'training': None, 'testing': None, 'validation': None}
            model.mse = {'training': None, 'testing': None, 'validation': None}
            model.rmse = {'training': None, 'testing': None, 'validation': None}
            model.molecules = {'training': [], 'testing': [], 'validation': []}

            model.model_sets = {'training': None, 'testing': None, 'validation': None}
            for ml_set in predictions_xml:
                ml_set_name = None
                correct_model = False
                for prop in ml_set:
                    label = prop.tag.split('}')[1]
                    value = prop.text
                    if label == 'Id':
                        ml_set_name = value
                    if label == 'ModelId':
                        if value == model.model_id:
                            correct_model = True
                    if label == 'Type' and correct_model:
                        model.model_sets[value] = ml_set_name

            for set_type, set_name in model.model_sets.items():
                if set_name is not None:
                    df = self.compound_data[['Compound Id', 'smiles']]
                    predictions_csv = predictions_dir + f'/{set_name}/values'
                    predictions_csv = pd.read_csv(predictions_csv, sep='\t')
                    predicted_data = self.__combine_dfs__(predictions_csv, df)
                    pred_column = list(predicted_data.columns.difference(['Compound Id', 'smiles']))[0]
                    predicted_data = predicted_data.dropna(subset=[pred_column])
                    predicted_data = predicted_data[['Compound Id', pred_column]]

                    actual_data = model.raw_data
                    act_column = model.target_name
                    actual_data = actual_data.dropna(subset=[act_column])
                    actual_data = actual_data[['Compound Id', 'smiles', act_column]]

                    combined_data = self.__combine_dfs__(predicted_data, actual_data)
                    combined_data = combined_data.dropna(subset=[pred_column, act_column])
                    combined_data.to_csv('dev.csv')

                    if len(combined_data) > 0:
                        r2 = r2_score(combined_data[act_column], combined_data[pred_column])
                        mse = mean_squared_error(combined_data[act_column], combined_data[pred_column])
                        rmse = np.sqrt(mean_squared_error(combined_data[act_column], combined_data[pred_column]))
                        model.n[set_type] = len(combined_data)
                        model.r2[set_type] = r2
                        model.mse[set_type] = mse
                        model.rmse[set_type] = rmse
                        model.molecules[set_type] = list(combined_data['smiles'])

            val_molecules = model.molecules['validation']
            train_molecules = model.molecules['training']

            # Logic to seperate data in test/train/val
            def __fetch_set__(smiles):
                if smiles in val_molecules:
                    return 'test'
                elif smiles in train_molecules:
                    return 'train'
                else:
                    return 'val'

            model.raw_data['in_set'] = model.raw_data['smiles'].apply(__fetch_set__)

    def cleanup_dir(self):
        if self.cleanup and self.zipped:
            shutil.rmtree(self.directory, ignore_errors=True)

    def to_neo4j(self):
        ModelOrOutputToNeo4j(qsar_obj=self, molecules_per_batch=self.molecules_per_batch, port=self.port,
                             username=self.username, password=self.password)


class QsarModel:

    def __init__(self, model_id, property_id, name, directory):
        self.model_id = model_id
        self.property_id = property_id
        self.name = name
        self.directory = directory
