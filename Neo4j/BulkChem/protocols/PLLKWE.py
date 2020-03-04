from math import ceil
from time import clock
import os
from py2neo import Node, Relationship, NodeMatcher, Graph
from core.barzilay_predict import barzilayPredict
from rdkit import Chem
import pandas as pd
from Neo4j.BulkChem.backends import get_testing_df, get_prop_from_smiles_df, __timer__


"""

pKa, LogS, LogP, Kow, Water-energy = PLLKWE

The point of this script to machine learning the properties listed above using Barzilay_Predict. We have not
looked into the predicting power or usability of the Barzilay scripts, perhaps we can analyze the data a different
way other than RSME and R^2. This scripts attempts to allow analysis of the scripts using Neo4j to visualize
the data. 

This script will ML the different properties (data in training directory as training files) onto the chemicals
found in the bulk chemicals. Then relate molecules that have extremely similar values, difference in 0.05 percent,
for a property. The idea is if they have a similar value for a property, they must be similar in a way. Even if 
properties do not make much physical sense, such as pKa for D.D.T., the computer might be able to find relationships a
person can not find. 

Then the real test is the number of hops needed for the shortest path between two molecules is the determination for
how similar they are. Direct connections mean very related, no paths at all mean absolutely no relationships.

"""

class __compare__:
    ##################################
    class __init_neo__:

        @staticmethod
        def __clean_up_columns__(df):
            cols = df.columns.tolist()
            cols = "$&lol$@".join(cols)
            cols = cols.lower()
            cols = cols.split('$&lol$@')
            df.columns = cols
            return df

        @staticmethod
        def __train__(df, data_type):
            barzilayPredict(target_label=data_type, dataset_type='regression', df_to_train=df,
                            train=True, predict=False)

        @staticmethod
        def __predict__(df, data_type):
            print('Predicting {}'.format(data_type))
            results = barzilayPredict(target_label=data_type, dataset_type='regression', df_to_predict=df,
                                      train=False, predict=True)
            df[data_type] = results.predicted_results_df[data_type]
            df[data_type] = round(df[data_type], 3)
            return df

        @staticmethod
        def __get_con_smiles_dict__(df, con_smiles):
            con_smiles_dict = df.loc[df['canonical_smiles'] == con_smiles].to_dict('records')[0]
            return con_smiles_dict

        def __insert_molecules__(self, molecule_dict):
            molecule = Node("molecule", **molecule_dict)
            self.graph.merge(molecule, 'molecule', 'chemical_name')

        def __init__(self, graph, *files):

            self.graph = graph
            data_types = ['pka', 'log_p', 'log_s', 'kow', 'water-energy']

            print('Organizing Files')
            temp_files = []
            for file in files:
                df = pd.read_csv(file)
                file = file.split('/')
                file = file[len(file)-1]
                df.to_csv('bulkchem_datafiles/temp/{}'.format(file), index=False)
                temp_files.append('bulkchem_datafiles/temp/{}'.format(file))
            files = temp_files

            print("Initializing Neo4j database")
            for data_type in data_types:
                print("Training to predict {}".format(data_type))
                training_df = pd.read_csv(
                    'bulkchem_datafiles/Training_datafiles/{}.csv'.format(data_type))  # Train
                self.__train__(training_df, data_type)
                for file in files:
                    raw_data = pd.read_csv(file)
                    raw_data = self.__clean_up_columns__(raw_data)  # Make columns lowercase
                    raw_data['canonical_smiles'] = raw_data['smiles'].map(
                        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))  # add con smiles
                    raw_data = self.__predict__(raw_data, data_type)  # Predict datatype
                    raw_data.to_csv(file, index=False)

            print('Inserting Molecules')
            for file in files:
                raw_data = pd.read_csv(file)
                raw_data = self.__clean_up_columns__(raw_data)  # Make columns lowercase
                raw_data['canonical_smiles'] = raw_data['smiles'].map(
                    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))  # add con smiles
                raw_data['canonical_smiles'].map(lambda x: self.__insert_molecules__(
                    self.__get_con_smiles_dict__(raw_data, x)))  # Insert molecules into neo
    ##################################

    def compare_score(self, testing_score, testing_node, current_score, current_node, datatype):
        sim_score = abs(testing_score - current_score)  # Determination and create relationships
        if 0.005 > sim_score:
            Rel = Relationship(current_node, 'similar_{}'.format(datatype), testing_node)
            self.tx.create(Rel)

    def __main__(self):

        data_types = ['pka', 'log_p', 'log_s', 'kow', 'water-energy']

        print('Finding Relationships...')
        for node in self.raw_molecule_nodes:
            time_for_batch = clock()
            self.counter += 1
            self.molecules_remaining = self.len_nodes - self.counter
            print("{0} molecules left to compare".format(self.molecules_remaining))  # Let user know amount left
            node_dict = dict(node)
            for i in range(self.splits):
                self.tx = self.graph.begin()
                if (i * self.max_nodes_in_ram + self.max_nodes_in_ram) <= self.counter:
                    pass
                elif i == self.splits - 1 and self.counter >= self.abs_upper_limit:
                    pass
                else:
                    testing_df = get_testing_df(i, self.raw_molecule_nodes, self.max_nodes_in_ram, self.counter)
                    testing_df.to_csv('test.csv', index=False)
                    for data_type in data_types:  # Generate relationships
                        testing_df['canonical_smiles'].map(lambda x: self.compare_score(
                            get_prop_from_smiles_df(testing_df, x, data_type),
                            get_prop_from_smiles_df(testing_df, x, 'Node'),
                            node_dict[data_type],
                            node, data_type
                        ))
                    self.tx.commit()  # Commit Relationships
            time_for_batch = clock() - time_for_batch
            time_left_secs, self.m, self.o, self.time_df = __timer__(self.o, self.m, self.counter, self.len_nodes,
                                                                     self.molecules_remaining,
                                                                     time_for_batch,
                                                                     self.time_df,
                                                                     clock() - self.run_time)

    def __init__(self, max_nodes_in_ram, *files):
        self.graph = Graph()
        self.max_nodes_in_ram = max_nodes_in_ram
        self.tx = None
        self.__init_neo__(self.graph, *files)

        matcher = NodeMatcher(self.graph)
        self.raw_molecule_nodes = matcher.match("molecule")  # Get nodes from neo

        self.run_time = clock()  # Declare variables for timer
        self.average_time = None
        self.o = None
        self.m = None
        self.time_df = pd.DataFrame(columns=['Molecules Remaining', 'Time needed (s)', 'Total Time passed (min)',
                                             'Predicted Time Left (min)'])

        self.len_nodes = len(self.raw_molecule_nodes)

        if self.len_nodes <= 0:
            raise Exception("There are no nodes in the database, strange...")

        if max_nodes_in_ram > self.len_nodes:  # Verify max nodes does not exceed number of nodes in database
            self.max_nodes_in_ram = self.len_nodes
        else:
            self.max_nodes_in_ram = max_nodes_in_ram

        self.splits = ceil(self.len_nodes / self.max_nodes_in_ram)  # Split data in neo4j database into chunks
        self.abs_upper_limit = len(self.raw_molecule_nodes) - (self.splits - 1) * self.max_nodes_in_ram
        self.counter = 0

        self.__main__()
