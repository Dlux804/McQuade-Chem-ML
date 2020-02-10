from time import clock
from math import ceil
from py2neo import Graph, Relationship, NodeMatcher
from rdkit import Chem, DataStructs
import pandas as pd

pd.options.mode.chained_assignment = None


class create_relationships:  # Class to generate the different relationship protocols

    def timer(self, time_for_batch):  # Keep track of long it will take to finish
        if self.average_time is None:
            self.average_time = time_for_batch
        else:
            self.average_time = (self.average_time * self.counter + time_for_batch)/self.counter
        return self.average_time * (len(self.raw_nodes)-self.counter)

    def bulk_to_bulk(self, testing_df, testing_smiles, current_node, relationship):  #Insert relationships
        testing_molecule = testing_df.loc[testing_df['canonical_smiles'] == testing_smiles].to_dict('records')[0]
        Rel = Relationship(testing_molecule['Node'], relationship, current_node,
                           rdkit_sim_score=testing_molecule['rdkit_sim_score'])
        self.tx.create(Rel)  # Merge relationship

    @staticmethod
    def compare_rdkit_score(testing_smiles, current_mol):  # Compare rdkit score
        testing_mol = Chem.MolFromSmiles(testing_smiles)
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        return sim_score

    def protocol_Default(self, node, node_dict, testing_df):  # Relate only using rdkit score
        self.tx = self.graph.begin()
        testing_df['rdkit_sim_score'] = testing_df['canonical_smiles'].map(
            lambda x: self.compare_rdkit_score(x, Chem.MolFromSmiles(node_dict['canonical_smiles'])))
        testing_df['canonical_smiles'].map(
            lambda x: self.bulk_to_bulk(testing_df, x, node, 'Rdkit_Sim_Score'))
        self.tx.commit()

    def __get_testing_df__(self, i):  # Get dataframe to compare molecule to
        lower_limit = i * self.max_nodes_in_ram + self.counter
        upper_limit = lower_limit + self.max_nodes_in_ram
        bulk_dicts = []
        counter = lower_limit
        for node in self.raw_nodes.skip(lower_limit):
            if counter == upper_limit:
                break
            molecule = (dict(node))
            molecule['Node'] = node
            bulk_dicts.append(molecule)
            counter = counter + 1
        return pd.DataFrame(bulk_dicts)

    def __main__(self):  # Main loop, compare all molecules to each other
        for node in self.raw_nodes:
            time_for_batch = clock()
            molecules_remaining = len(self.raw_nodes) - self.counter
            self.counter = self.counter + 1
            print("{0} molecules left to compare".format(molecules_remaining))  # Let user know amount left
            node_dict = (dict(node))
            for i in range(self.splits):
                if (i * self.max_nodes_in_ram + self.max_nodes_in_ram) <= self.counter:
                    pass
                elif i == self.splits-1 and self.counter >= self.abs_upper_limit:
                    pass
                else:
                    testing_df = self.__get_testing_df__(i)
                    self.protocol_dict[self.protocol](node, node_dict, testing_df)
            time_left_minutes = round(self.timer(time_for_batch) / 60, 2)
            time_left_hours = round(time_left_minutes / 60, 2)
            self.time_df = self.time_df.append({'Molecules Remaining': molecules_remaining, 'Time needed (s)': time_for_batch,
                                      'Total Time passed (min)': self.run_time}, ignore_index=True)
            print("\nTime Remaining: {0} minutes ({1} hours)".format(time_left_minutes, time_left_hours))
        self.time_df.to_csv('Time_vs_molecules.csv', index=False)

    def __init__(self, protocol, max_nodes_in_ram=3162):

        self.run_time = clock()  # Declare variables for timer
        self.average_time = None
        self.n = 0
        self.time_df = pd.DataFrame(columns=['Molecules Remaining', 'Time needed (s)', 'Total Time passed (min)'])

        self.graph = Graph()  # Get neo4j graph database
        self.tx = None

        self.protocol = protocol  # Define protocols
        self.protocol_dict = {
            "Default": self.protocol_Default,
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        matcher = NodeMatcher(self.graph)  # Get nodes from neo4j database (Doesn't load them in ram yet)
        self.raw_nodes = matcher.match("bulkChemMolecule")

        if max_nodes_in_ram > len(self.raw_nodes):
            self.max_nodes_in_ram = len(self.raw_nodes)
        else:
            self.max_nodes_in_ram = max_nodes_in_ram  # Split data in neo4j database into chunks

        self.splits = ceil(len(self.raw_nodes) / self.max_nodes_in_ram)
        self.abs_upper_limit = len(self.raw_nodes) - (self.splits-1)*self.max_nodes_in_ram
        self.counter = 0

        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))  # Start comparing
        self.__main__()