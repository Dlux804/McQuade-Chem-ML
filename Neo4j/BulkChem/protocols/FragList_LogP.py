from math import ceil
from time import clock
from py2neo import Node, Relationship, NodeMatcher, Graph
from core.barzilay_predict import barzilayPredict
from rdkit import Chem, DataStructs
import pandas as pd
from Neo4j.BulkChem.backends import get_testing_df, get_fragments, get_prop_from_smiles_df, __timer__


class __compare__:
    ##################################
    class __init_neo__:  # TODO find a way to separate __init_neo__ in a way that makes sense

        @staticmethod
        def __clean_up_columns__(df):
            cols = df.columns.tolist()
            cols = "$&lol$@".join(cols)
            cols = cols.lower()
            cols = cols.split('$&lol$@')
            df.columns = cols
            return df

        @staticmethod
        def __predict_logP__(df):
            print('Predicting LogP')
            results = barzilayPredict(target_label='exp', dataset_type='regression', df_to_predict=df,
                                      train=False, predict=True)
            df['logp'] = results.predicted_results_df['exp']
            df['logp'] = round(df['logp'], 3)
            return df

        @staticmethod
        def __get_con_smiles_dict__(df, con_smiles):  # TODO look for a better way to get properties from DataFrame
            con_smiles_dict = df.loc[df['canonical_smiles'] == con_smiles].to_dict('records')[0]
            return con_smiles_dict

        def __insert_molecules__(self, molecule_dict):
            molecule = Node("molecule", chemical_name=molecule_dict['product'],
                            cas=molecule_dict['cas'],
                            smiles=molecule_dict['smiles'],
                            canonical_smiles=molecule_dict['canonical_smiles'],
                            logp=molecule_dict['logp'],
                            fragments=molecule_dict['fragments'])
            self.graph.merge(molecule, 'molecule', 'chemical_name')

        def __init__(self, graph, *files):
            print("Initializing Neo4j database")
            print("Training to predict LogP")
            training_df = pd.read_csv('bulkchem_datafiles/Lipophilicity-ID.csv')  # Train to from LogP data
            barzilayPredict(target_label='exp', dataset_type='regression', df_to_train=training_df,
                            train=True, predict=False)
            self.graph = graph
            for file in files:
                raw_data = pd.read_csv(file)
                raw_data = self.__clean_up_columns__(raw_data)  # Make columns lowercase
                raw_data['canonical_smiles'] = raw_data['smiles'].map(  # add con smiles
                    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
                raw_data = self.__predict_logP__(raw_data)  # Predict logP
                print("Calculating Fragments for file: {}".format(file))
                raw_data['fragments'] = raw_data['canonical_smiles'].map(lambda x: get_fragments(x))  # add fragments
                print("Inserting Molecules for file {}".format(file))
                raw_data['canonical_smiles'].map(lambda x: self.__insert_molecules__(  # Insert molecules into neo
                    self.__get_con_smiles_dict__(raw_data, x)
                ))
                matcher = NodeMatcher(self.graph)
                self.raw_molecule_nodes = matcher.match("molecule")  # Get nodes from neo
    ##################################

    def insert_fragment_list_nodes(self):
        print('Inserting fragment(s) nodes before comparing molecules...')
        fragments = open('bulkchem_datafiles/List_of_fragment_groups.txt', 'r')
        fragments = fragments.readlines()[0].split(', ')
        for i in range(len(fragments)):
            fragment_list = []
            for x in range(i):
                if not fragment_list:
                    fragment_list = [fragments[i]]
                    fragment_list_string = ', '.join(fragment_list)
                    fragment_list_Node = Node("fragments", fragment_list=fragment_list_string)
                    self.graph.merge(fragment_list_Node, 'fragments', 'fragment_list')
                fragment_list.append(fragments[x])
                fragment_list_string = ', '.join(fragment_list)
                fragment_list_Node = Node("fragments", fragment_list=fragment_list_string)
                self.graph.merge(fragment_list_Node, 'fragments', 'fragment_list')
        print('Done')
        matcher = NodeMatcher(self.graph)
        return matcher.match("fragments")

    def __fragment_list_comparision__(self, current_node, current_node_dict):
        current_frags = str(current_node_dict['fragments'])
        current_frags = current_frags.split(", ")
        for node in self.raw_fragment_nodes:
            node_dict = dict(node)
            node_frags = node_dict['fragment_list'].split(", ")
            comparison_1 = list(
                set(current_frags) - {i for e in node_frags for i in current_frags if e in i})
            comparison_2 = list(
                set(node_frags) - {i for e in current_frags for i in node_frags if e in i})
            if not comparison_1 and not comparison_2:
                Rel = Relationship(current_node, 'has_fragment_list', node)
                self.tx = self.graph.begin()
                self.tx.create(Rel)
                self.tx.commit()
                break

    def compare_logp_score(self, testing_logp, testing_node, current_logp, current_node):
        sim_score = abs(testing_logp - current_logp)
        if 0.001 > sim_score:
            Rel = Relationship(current_node, 'similar_logp', testing_node)
            self.tx.create(Rel)

    def __main__(self):
        print('Finding Relationships...')
        for node in self.raw_molecule_nodes:
            time_for_batch = clock()
            self.counter += 1
            self.molecules_remaining = self.len_nodes - self.counter
            print("{0} molecules left to compare".format(self.molecules_remaining))  # Let user know amount left
            node_dict = dict(node)
            self.__fragment_list_comparision__(node, node_dict)
            for i in range(self.splits):
                self.tx = self.graph.begin()
                if (i * self.max_nodes_in_ram + self.max_nodes_in_ram) <= self.counter:
                    pass
                elif i == self.splits - 1 and self.counter >= self.abs_upper_limit:
                    pass
                else:
                    testing_df = get_testing_df(i, self.raw_molecule_nodes, self.max_nodes_in_ram, self.counter)
                    testing_df.to_csv('test.csv')
                    testing_df['canonical_smiles'].map(lambda x: self.compare_logp_score(
                        get_prop_from_smiles_df(testing_df, x, 'logp'),
                        get_prop_from_smiles_df(testing_df, x, 'Node'),
                        node_dict['logp'],
                        node
                    ))
                self.tx.commit()
            time_for_batch = clock() - time_for_batch
            time_left_secs, self.m, self.o, self.time_df = __timer__(self.o, self.m, self.counter, self.len_nodes,
                                                                     self.molecules_remaining,
                                                                     time_for_batch,
                                                                     self.time_df,
                                                                     self.run_time)

    def __init__(self, max_nodes_in_ram, *files):
        self.graph = Graph()
        self.max_nodes_in_ram = max_nodes_in_ram
        self.tx = None
        self.raw_molecule_nodes = self.__init_neo__(self.graph, *files).raw_molecule_nodes
        self.raw_fragment_nodes = self.insert_fragment_list_nodes()

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
