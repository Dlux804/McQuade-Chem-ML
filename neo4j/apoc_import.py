import pandas as pd
from core import models, ingest
from core.misc import cd
import os
import csv

"""

Create a class for neo4j datasets.

Takes as input the file (likely csv), and a tuple containing the column header of the measurement and 
what the measurement is.  i.e ('expt', logP)

Use the dataset class to create the node file, with optional header file.  Basically done already.
Use the dataset class to create realationship files for the datasets.  Dataset --[CONTAINS]-> Molecule

Expected challenges:  I could make header files for all molsets because they will have the same features.  Only issue
is that the measurement columns are called separate things.  I would have to add in a section to rename the exp column.
Which wouldn't be bad. 

Need to create formatted csv files for nodes and relationships for use in apoc.import.csv

Header for node would look something like this:
smiles:ID(Molecule-ID),name:STRING,MolWt:INT, logP:FLOAT

Header for relationship file would look like this
:START_ID(Molecule-ID),:END_ID(Molecule-ID),solubility:FLOAT

I also want to output an auto generated Cypher command for uploading the data.  

"""

# define datasets in dict.
# key is file name
# value is tuple of target column name and measured property
sets = {
        # 'Lipophilicity-ID.csv': ('exp', 'logP'),
        'ESOL.csv': ('water-sol', 'logS'),
        # 'water-energy.csv': ('expt', 'hydration_energy')
        'logP14k.csv': ('Kow', 'logP'),
        # 'jak2_pic50.csv': ('pIC50', 'pIC50'),
        # 'flashpoint2.csv': ('flashpoint', 'flash_point')
    }

class dataset:
    """
    Class for preparing data to be imported to Neo4j via apoc.import.csv .
    """
    def __init__(self, file, measurement):
        """Requires: filename of csv and tuple of measurment column and property"""
        self.file = file  # filename
        self.measurement = measurement[1]  # what is the measured property in the dataset
        self.target = measurement[0]  # column name of measured property in csv
        print('Initialized {} dataset.'.format(self.file))


    def enrich(self, feat_meth=[0]):
        """
        expand dataset by calculating properties of molecules.
        Feat_meth determines what featurization method is used for calculating properties
        """
        alg = 'rf'  # not actually used, but necessary

        # change active directory
        with cd('../dataFiles'):  # assumes data file location

            # initiate model class with algorithm, dataset and target
            model = models.MlModel(alg, self.file, self.target)

            # featurize molecules
            model.featurization(feat_meth)

            model.data['exp'] = model.data[self.target]  # new column with measurment data
            model.data = model.data.drop(columns=['RDKit2D_calculated', self.target], axis=1)  # drop old target
        self.data = model.data  # save dataframe to instance

    def mol_nodes(self):
        """
        Make the nodes file for apoc.import.csv
        """
        # dictionary for converting between numpy and neo4j data types
        type_dict = {
            'object': ':STRING',  # strings are stored as objects in numpy
            'float64': ':FLOAT',
            'int64': ':INT',
            'bool': ':BOOLEAN'
        }
        types_dict = self.data.dtypes.to_dict()  # get data type of each column in dict
        cols = list(self.data.columns)  # get column headers
        header = ['smiles:ID']
        cols = cols[1:]  # get rid of smiles column header.  Done explicitly
        for col in cols:
            head = col + type_dict[types_dict[col].name]
            header.append(head)
        self.mol_nodes = self.data
        self.mol_nodes.columns = header
        # print(self.nodes_file.head(5))
        self.mol_node_file = 'mol-nodes-'+self.file
        self.mol_nodes.to_csv(self.mol_node_file, index=False)


    def set_nodes(self):
        """Make node file for dataset."""
        with cd('../dataFiles'):  # assumes data file location
            with open(self.file) as csvfile:
                fileObject = csv.reader(csvfile)
                # for row in fileObject:
                #     print(row)
                row_count = sum(1 for row in fileObject) - 1  # count number entities
        # print("row count:", row_count)
        data = {'prop': [self.measurement], 'title:ID': [self.file], 'size:INT': [row_count]}
        # print(data)

        self.set_node = pd.DataFrame.from_dict(data)
        # print(self.set_node)
        self.set_node_file = 'set-node-' + self.file
        self.set_node.to_csv(self.set_node_file, index=False)


    def rels(self):
        """ make the relationship file for apoc.import.csv"""
        with cd('../dataFiles'):  # assumes data file location
            # rels = pd.read_csv(self.file)
            rels = ingest.load_smiles(self.file, self.target)[0]
            print(rels.head(5))

        rels[':START_ID'] = self.file  # add relationship starting point as dataset file name
        rels = rels.rename(columns={'smiles': ':END_ID'})  # rename column

        # drop every column but start and end ID
        self.rels = rels[[':START_ID', ':END_ID']]

        # write to file
        self.rels_file = 'rels-' + self.file
        self.rels.to_csv(self.rels_file, index=False)


    def apoc_cmd(self):
        """Generate a Cypher command for loading the files created with this class."""
        print()
        print("CALL apoc.import.csv(\n[{{fileName: 'file:/{}', labels: ['Molecule']}}, {{fileName: 'file:/{}', labels: ['Dataset']}}],\n[{{fileName: 'file:/{}', type: 'CONTAINS'}}],\n{{}})".format(self.mol_node_file, self.set_node_file, self.rels_file))



set1 = dataset('ESOL.csv', sets['ESOL.csv'])
set1.enrich()
set1.mol_nodes()
set1.set_nodes()
set1.rels()
set1.apoc_cmd()