import pandas as pd
import ast
import os
import timeit
from math import ceil

from rdkit.Chem import MolFromSmiles, MolToSmiles
from py2neo import Node, NodeMatcher, Graph
from Neo4j.US_patents import csv_to_neo4j_backends


class file_to_neo4j:

    def __init__(self, file, max_nodes_in_ram, ignore_checker_files=False):

        self.file = file
        if not ignore_checker_files:
            if os.path.exists(self.file + ".checker"):
                return

        self.raw_data = pd.read_csv(file)
        self.graph = Graph()

        matcher = NodeMatcher(self.graph)
        self.in_database_compounds = matcher.match("compound")
        self.len_nodes = len(self.in_database_compounds)
        self.max_nodes_in_ram = max_nodes_in_ram
        self.splits = ceil(self.len_nodes / self.max_nodes_in_ram)

        print("\n-----------------------------------------------------------------------------------------------------")
        self.timer = timeit.default_timer()
        print("The file being inserted is {}".format(self.file))
        self.csv_file_compounds = self.gather_compounds()
        print("There are {} unfiltered compounds in csv file".format(len(self.csv_file_compounds)))
        print("Comparing compounds in csv file to compounds in database")
        print("The numbers of chunks to compare against is {}, max nodes in ram allowed is {}, "
              "there are {} compounds in the database".format(self.splits, self.max_nodes_in_ram, self.len_nodes))
        self.csv_file_compounds = self.search_for_existing_nodes()
        print("Comparing csv compounds to csv compounds, creating nodes for new unique compounds")
        self.smiles_node_dicts = self.gather_all_compounds_without_matches_nodes()
        print("Adding nodes to csv compounds")
        self.csv_file_compounds = self.add_nodes_to_new_compounds()
        print("Finalizing csv compounds, committing reactions to database")
        self.final_df = self.clean_raw_data()
        csv_to_neo4j_backends.data_to_neo4j(self.final_df)
        open(self.file + ".checker", "a").close()
        self.time_needed_to_insert = round((timeit.default_timer() - self.timer)/60, 2)
        print("Time needed {} minutes".format(self.time_needed_to_insert))
        print("-----------------------------------------------------------------------------------------------------\n")

    def gather_compounds(self):
        all_compounds = []
        compound_label_names = ['reactants', 'products', 'solvents', 'catalyst']
        for index, row in self.raw_data.iterrows():
            row_dict = dict(row)
            for item in row_dict:
                if item in compound_label_names:
                    try:
                        compounds = ast.literal_eval(row_dict[item])
                    except ValueError:
                        compounds = row_dict[item]
                    except SyntaxError:
                        compounds = row_dict[item]
                    for compound in compounds:
                        identifiers = compound['identifiers']
                        for identifier in identifiers:
                            try:
                                identifier_name = identifier.split(' = ')[0]
                                identifier_value = identifier.split(' = ')[1]
                            except IndexError:
                                identifier_name = identifier.split(':')[0]
                                identifier_value = identifier.split(':')[1]
                            if identifier_name == 'smiles':
                                try:
                                    mol = MolFromSmiles(identifier_value)
                                    identifier_value = MolToSmiles(mol)
                                except:
                                    break
                            compound[identifier_name] = identifier_value
                        compound['node'] = None
                        if identifiers:
                            compound.pop('identifiers')
                        if 'smiles' in compound.keys():
                            all_compounds.append(compound)
        all_compounds_df = pd.DataFrame.from_records(all_compounds)
        return all_compounds_df

    def get_current_nodes_df(self, i):
        batch_nodes_dicts = []
        lower_cap = i * self.max_nodes_in_ram
        upper_cap = lower_cap + self.max_nodes_in_ram
        print("Gathering nodes from Neo4j, currently on chunk {} of {}".format(i+1, self.splits))
        nodes = self.in_database_compounds.skip(lower_cap)
        counter = lower_cap
        for node in nodes:
            batch_node_dict = {'node_ID': node.identity, 'smiles': dict(node)['smiles'], 'node': node}
            batch_nodes_dicts.append(batch_node_dict)
            counter = counter + 1
            if counter == upper_cap or counter == self.len_nodes:
                return batch_nodes_dicts

    @staticmethod
    def search_for_matching_node(batch_nodes_dicts, compound):
        if 'smiles' not in compound.keys():
            return None
        smiles = compound['smiles']
        for smiles_node_dict in batch_nodes_dicts:
            node_matched_smiles = smiles_node_dict['smiles']
            if smiles == node_matched_smiles:
                return smiles_node_dict['node']
        return None

    def update_raw_data_df(self, batch_nodes_dicts):
        row_dicts = []
        len_csv_compounds = len(self.csv_file_compounds)
        timer_counter = 0
        i = 0
        for index, row in self.csv_file_compounds.iterrows():
            if timer_counter >= len_csv_compounds/10:
                percent_compared = round(i/len_csv_compounds*100, 0)
                print("Percentage of csv compounds compared in chunk {}%".format(percent_compared))
                timer_counter = 0
            compound = dict(row)
            match = self.search_for_matching_node(batch_nodes_dicts, compound)
            if match is not None:
                compound['node'] = match
            row_dicts.append(compound)
            i = i + 1
            timer_counter = timer_counter + 1
        return pd.DataFrame.from_records(row_dicts)

    def search_for_existing_nodes(self):
        csv_file_compounds = self.csv_file_compounds
        for i in range(self.splits):
            batch_nodes_dicts = self.get_current_nodes_df(i)
            print("Comparing csv compounds to compounds in chunk from Neo4j")
            csv_file_compounds = self.update_raw_data_df(batch_nodes_dicts)
        return csv_file_compounds

    def gather_all_compounds_without_matches_nodes(self):
        filtered_df = self.csv_file_compounds[self.csv_file_compounds['node'].isnull()]
        filtered_df = filtered_df.drop_duplicates(subset=['smiles'])
        smiles_node_dicts = []
        for index, row in filtered_df.iterrows():
            row_dict = dict(row)
            node = Node('compound', **row_dict)
            smiles = row['smiles']
            row_dict.pop('node')
            tx = self.graph.begin()
            tx.create(node)
            tx.commit()
            matcher = NodeMatcher(self.graph)
            node = matcher.get(node.identity)
            smiles_node_dicts.append({'smiles': smiles, 'node': node})
        return smiles_node_dicts

    def add_nodes_to_new_compounds(self):
        row_dicts = []
        for index, row in self.csv_file_compounds.iterrows():
            row_dict = dict(row)
            if str(row_dict['node']) == 'nan' or row_dict['node'] is None:
                smiles = row_dict['smiles']
                for smiles_node_dict in self.smiles_node_dicts:
                    node_matched_smiles = smiles_node_dict['smiles']
                    if smiles == node_matched_smiles:
                        row_dict['node'] = smiles_node_dict['node']
            row_dicts.append(row_dict)
        cleaned_compounds_df = pd.DataFrame.from_records(row_dicts)
        cleaned_compounds_df = cleaned_compounds_df.drop_duplicates(subset=['smiles'])
        return cleaned_compounds_df

    def clean_raw_data(self):
        all_compounds = []
        compound_label_names = ['reactants', 'products', 'solvents', 'catalyst']
        for index, row in self.raw_data.iterrows():
            row_dict = dict(row)
            for item in row_dict:
                if item in compound_label_names:
                    nodes = []
                    try:
                        compounds = ast.literal_eval(row_dict[item])
                    except TypeError:
                        compounds = row_dict[item]
                    except ValueError:
                        compounds = row_dict[item]
                    except SyntaxError:
                        compounds = row_dict[item]
                    for compound in compounds:
                        identifiers = compound['identifiers']
                        for identifier in identifiers:
                            try:
                                identifier_name = identifier.split(' = ')[0]
                                identifier_value = identifier.split(' = ')[1]
                            except IndexError:
                                identifier_name = identifier.split(':')[0]
                                identifier_value = identifier.split(':')[1]
                            if identifier_name == 'smiles':
                                try:
                                    mol = MolFromSmiles(identifier_value)
                                    smiles = MolToSmiles(mol)
                                except:
                                    break
                                match = self.csv_file_compounds.loc[self.csv_file_compounds['smiles'] == smiles]
                                match = match.to_dict('records')[0]
                                nodes.append(match['node'])
                    row_dict[item] = nodes
            all_compounds.append(row_dict)
        all_compounds_df = pd.DataFrame.from_records(all_compounds)
        return all_compounds_df
