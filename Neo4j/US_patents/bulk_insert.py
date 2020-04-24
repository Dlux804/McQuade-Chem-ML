import pandas as pd
import ast
import os
import timeit

from rdkit.Chem import MolFromSmiles, MolToSmiles
from py2neo import Node, NodeMatcher, Graph
from Neo4j.US_patents import csv_to_neo4j_backends


class US_patents_to_neo:

    def __init__(self, main_directory):

        if not os.path.exists('temp_files'):
            os.mkdir('temp_files')

        self.main_directory = main_directory
        self.graph = Graph()
        print('\n-----------------------------------------------------------------------------------------------------')
        print("Safe to pause script and resume later")
        print('-----------------------------------------------------------------------------------------------------\n')
        print("Gathering and organizing compounds from the csv files")
        self.patent_to_csv()
        print('\n-----------------------------------------------------------------------------------------------------')
        print("NOT SAFE TO PAUSE SCRIPT, please let run or risk breaking code")
        print('-----------------------------------------------------------------------------------------------------\n')
        print("Comparing large csv files, dropping duplicates")
        self.compare_csvs_to_csvs()
        print("Combining csv files into one large file")
        self.combine_csv_files()
        print("Inserting nodes in Neo4j and indexing compounds")
        self.insert_and_index_compounds()
        print('\n-----------------------------------------------------------------------------------------------------')
        print('Safe again to pause script and resume later')
        print('-----------------------------------------------------------------------------------------------------\n')
        print("Loading very large csv file into memory")
        self.unique_compound_data = pd.read_csv('temp_files/unique_compound_data.csv')
        print("Matching Nodes and Compounds in original csv files")
        if not os.path.exists('time_df.csv'):
            time_df = pd.DataFrame(columns=['directory',
                                            'number of reactions in csv file', 'time needed to insert (min)',])
            time_df.to_csv('time_df.csv', index=False)
        self.patent_to_csv(nodes_inserted=True)

    @staticmethod
    def csv_to_compounds(csv_df):
        all_compounds = []
        compound_label_names = ['reactants', 'products', 'solvents', 'catalyst']
        for index, row in csv_df.iterrows():
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
                            identifier_name = identifier.split(' = ')[0]
                            identifier_value = identifier.split(' = ')[1]
                            if identifier_name == 'smiles':
                                try:
                                    mol = MolFromSmiles(identifier_value)
                                    identifier_value = MolToSmiles(mol)
                                except:
                                    break
                            compound[identifier_name] = identifier_value
                        if identifiers:
                            compound.pop('identifiers')
                        try:
                            compound.pop('amounts')
                        except KeyError:
                            pass
                        if 'smiles' in compound.keys():
                            all_compounds.append({'smiles': compound['smiles'], 'compound': compound})
        return all_compounds

    def patent_to_csv(self, nodes_inserted=False):
        counter = 0
        main_directories = os.listdir(self.main_directory)
        if not nodes_inserted and os.path.exists('temp_files/unique_compound_data.csv'):
            return
        for main_directory in main_directories:
            main_directory = self.main_directory + "/" + main_directory
            for directory in os.listdir(main_directory):
                if len(directory.split('_')) != 1:
                    directory = main_directory + "/" + directory
                    print(directory)
                    if not os.path.exists('temp_files/{}.csv'.format(counter)) or nodes_inserted:
                        all_compounds_dir = []
                        sub_counter = 0
                        for file in os.listdir(directory):
                            file = directory + "/" + file
                            try:
                                if nodes_inserted and not os.path.exists(file + '.checker'):
                                    timer = timeit.default_timer()
                                    number_of_files_in_dir = 0
                                    for testing_file in os.listdir(directory):
                                        file_ext = testing_file.split('.')
                                        if file_ext[len(file_ext) - 1] != 'checker':
                                            number_of_files_in_dir = number_of_files_in_dir + 1
                                    raw_csv_file_data = pd.read_csv(file)
                                    print("{} files left to insert into directory"
                                          .format(number_of_files_in_dir - sub_counter))
                                    self.insert_csv_to_neo4j(self.unique_compound_data, raw_csv_file_data)
                                    time_needed_to_insert = round((timeit.default_timer() - timer) / 60, 2)
                                    time_df = pd.read_csv('time_df.csv')
                                    time_df = time_df.append({'directory': directory,
                                                              'number of reactions in csv file': len(raw_csv_file_data),
                                                              'time needed to insert (min)': time_needed_to_insert},
                                                             ignore_index=True)
                                    time_df.to_csv('time_df.csv', index=False)
                                    open(file + ".checker", "a").close()
                                    print("Time needed {} minutes".format(time_needed_to_insert))
                                elif nodes_inserted:
                                    pass
                                else:
                                    raw_csv_file_data = pd.read_csv(file)
                                    csv_compounds = self.csv_to_compounds(raw_csv_file_data)
                                    all_compounds_dir.extend(csv_compounds)
                            except pd.errors.EmptyDataError:
                                pass
                            file_ext = file.split('.')
                            if file_ext[len(file_ext) - 1] != 'checker':
                                sub_counter = sub_counter + 1
                        if not nodes_inserted:
                            dir_compounds_df = pd.DataFrame.from_records(all_compounds_dir)
                            dir_compounds_df = dir_compounds_df.drop_duplicates(subset=['smiles'])
                            dir_compounds_df.to_csv('temp_files/{}.csv'.format(counter), index=False)
                        counter = counter + 1
                    else:
                        counter = counter + 1

    @staticmethod
    def compare_csvs_to_csvs():
        files = os.listdir('temp_files')
        number_of_files = len(files)
        number_of_comparisons_needed = int((number_of_files + number_of_files ** 2) / 2 - number_of_files)
        i = 0
        sub_counter = 0
        counter = 0
        while i < number_of_files:
            main_file = 'temp_files/{}.csv'.format(i)
            j = i + 1
            while j < number_of_files:
                if sub_counter >= number_of_comparisons_needed / 25:
                    percent_complete = round(counter / number_of_comparisons_needed * 100, 1)
                    print("Percent complete comparing files {}%".format(percent_complete))
                    sub_counter = 0
                sub_file = 'temp_files/{}.csv'.format(j)
                df1 = pd.read_csv(main_file)
                df1['FileName'] = 'df1'
                df2 = pd.read_csv(sub_file)
                df2['FileName'] = 'df2'
                final = df1.append(df2, ignore_index=True)
                final = final.drop_duplicates(subset=['smiles'])
                final = final.loc[final['FileName'] == 'df2']
                final.drop(columns=['FileName'])
                final.to_csv(sub_file, index=False)
                sub_counter = sub_counter + 1
                counter = counter + 1
                j = j + 1
            i = i + 1

    @staticmethod
    def combine_csv_files():
        if len(os.listdir('temp_files')) > 2:
            main_data = pd.read_csv('temp_files/0.csv')
            for file in os.listdir('temp_files'):
                file = 'temp_files/' + file
                if file != 'temp_files/0.csv' and file != 'temp_files/unique_compound_data.csv':
                    sub_data = pd.read_csv(file)
                    main_data = main_data.append(sub_data, ignore_index=True)
            try:
                main_data.to_csv('temp_files/unique_compound_data.csv', index=False, sort=True)
            except TypeError:
                main_data.to_csv('temp_files/unique_compound_data.csv', index=False)
        elif len(os.listdir('temp_files')) == 1:
            for file in os.listdir('temp_files'):
                file = 'temp_files/' + file
                if file != 'temp_files/unique_compound_data.csv':
                    main_data = pd.read_csv('temp_files/0.csv')
                    try:
                        main_data.to_csv('temp_files/unique_compound_data.csv', index=False, sort=True)
                    except TypeError:
                        main_data.to_csv('temp_files/unique_compound_data.csv', index=False)
        for file in os.listdir('temp_files'):
            file = 'temp_files/' + file
            if file != 'temp_files/unique_compound_data.csv':
                os.remove(file)

    def insert_and_index_compounds(self):

        test_data = pd.read_csv('temp_files/unique_compound_data.csv', nrows=1)
        for index, row in test_data.iterrows():
            row_dict = dict(row)
            try:
                ast.literal_eval(row_dict['compound'])
            except KeyError:
                return

        bulk_csv_data = pd.read_csv('temp_files/unique_compound_data.csv')
        smiles_node_dicts = []
        number_of_compounds = len(bulk_csv_data)
        counter = 0
        sub_counter = 0
        print("There are {} unique compounds to insert and index".format(number_of_compounds))
        for index, row in bulk_csv_data.iterrows():
            if sub_counter >= number_of_compounds / 100:
                percent_complete = round(counter / number_of_compounds * 100, 1)
                print("Percent complete inserting and indexing file {}%".format(percent_complete))
                sub_counter = 0
            row_dict = dict(row)
            compound = ast.literal_eval(row_dict['compound'])
            node = Node('compound', **compound)
            smiles = row['smiles']
            tx = self.graph.begin()
            tx.create(node)
            tx.commit()
            matcher = NodeMatcher(self.graph)
            node = matcher.get(node.identity)
            smiles_node_dicts.append({'node': node.identity, 'smiles': smiles, 'node_dict': compound})
            counter = counter + 1
            sub_counter = sub_counter + 1
        indexed_compounds = pd.DataFrame.from_records(smiles_node_dicts)
        indexed_compounds.to_csv('temp_files/unique_compound_data.csv', index=False)

    @staticmethod
    def match_compound(unique_compound_data, smiles):
        df = unique_compound_data
        match = df.loc[df['smiles'] == smiles]
        if len(match) > 0:
            match = match.to_dict('records')[0]
            return match['node']
        else:
            print('{}: Something odd is afloat...'.format(smiles))

    def insert_csv_to_neo4j(self, unique_compound_data, csv_df):
        all_compounds = []
        compound_label_names = ['reactants', 'products', 'solvents', 'catalyst']
        for index, row in csv_df.iterrows():
            row_dict = dict(row)
            for item in row_dict:
                if item in compound_label_names:
                    nodes = []
                    try:
                        compounds = ast.literal_eval(row_dict[item])
                    except ValueError:
                        compounds = row_dict[item]
                    except SyntaxError:
                        compounds = row_dict[item]
                    for compound in compounds:
                        identifiers = compound['identifiers']
                        for identifier in identifiers:
                            identifier_name = identifier.split(' = ')[0]
                            identifier_value = identifier.split(' = ')[1]
                            if identifier_name == 'smiles':
                                try:
                                    mol = MolFromSmiles(identifier_value)
                                    identifier_value = MolToSmiles(mol)
                                except:
                                    break
                            compound[identifier_name] = identifier_value
                        if identifiers:
                            compound.pop('identifiers')
                        try:
                            compound.pop('amounts')
                        except KeyError:
                            pass
                        if 'smiles' in compound.keys():
                            matched_compound = self.match_compound(unique_compound_data, compound['smiles'])
                            nodes.append(matched_compound)
                    row_dict[item] = nodes
            all_compounds.append(row_dict)
        csv_df = pd.DataFrame.from_records(all_compounds)
        csv_to_neo4j_backends.data_to_neo4j(csv_df)
