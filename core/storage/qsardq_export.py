import os
import re
import shutil
from lxml import etree
from rdkit.Chem import MolFromSmiles, MolToMolBlock
import xml.etree.cElementTree as ET
from sklearn2pmml import PMMLPipeline, sklearn2pmml


def QsarDB_export(self, zip_output=False):
    """
    :param zip_output: Weather to zip output folder
    :param self: Model Object
    :return:

    Purpose of this is to export our models in a format that complies with the QsarDB requirements.
    This script is meant to be run on a model that has been fully featurized and tested.
    """

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
        sklearn2pmml(pipeline, "pmml.xml", with_repr=True)

        # Read in the file
        with open('pmml.xml', 'r') as file:
            filedata = file.read()

        # Replace x[1-...] with actual column names
        m = re.findall('x\-?\d+', filedata)
        matches = []
        [matches.append(x) for x in m if x not in matches]
        feature_cols = list(data_df.columns.difference(["in_set", "smiles", "id", self.target_name]))
        matched_dict = dict(zip(matches, feature_cols))
        for match, feat in matched_dict.items():
            filedata = filedata.replace(match, f'{feat}')

        # Replace y with target name
        filedata = filedata.replace('y', f'{self.target_name}')

        # Write the file out again
        with open('pmml.xml', 'w') as file:
            file.write(filedata)

    # Get current directory, change in qdb directory
    cur_dir = __mkd__(f'{self.run_name}_qdb')
    non_descriptors_columns = ['smiles', 'id', 'in_set', self.target_name]

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
    ET.SubElement(model, "Cargos").text = "pmml.xml bibtex"
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
