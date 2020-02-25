from time import clock
from math import ceil
from py2neo import Graph, Relationship, NodeMatcher
from rdkit import Chem, DataStructs
import pandas as pd

pd.options.mode.chained_assignment = None  # Help speed up code and drop weird warning pandas can give


class create_relationships:  # Class to generate the different relationship protocols

    """
    Protocol Rdkit_sim_score is a relationship protocol that will relate EVERY molecule to EVERY OTHER molecule
    with a relationship labeled 'rdkit_sim_score'. Each rdkit_sim_score relationship has an attribute with the score
    that is given when the similarity test is run comparing two molecules. This can be useful because it takes time to
    calculate the rdkit similarity score, and quickly telling what molecules are similar to other ones can be useful.
    There is an fundamental issue with this, however, which is that there are a lot of operations that need to be
    writen. Since there is a relationship relating every molecule to every other molecule by definition, and there
    are ~3000 molecules in the bulkchem data, that is 3000+2999+2998+...+1 relationships
    (The general formula for this is n(n+1)/2, meaning there are ~4.5 million relationships). Efficiency is needed,
    this file attempts to increase the efficiency of inserting bulk relationships
    """

    @staticmethod
    def compare_rdkit_score(testing_smiles, current_mol):  # Compare rdkit score
        testing_mol = Chem.MolFromSmiles(testing_smiles)
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        return sim_score

    def protocol_Rdkit_sim_score(self, node, node_dict):  # Relate only using rdkit score
        self.tx = self.graph.begin()
        self.testing_df['rdkit_sim_score'] = self.testing_df['canonical_smiles'].map(
            lambda x: self.compare_rdkit_score(x, Chem.MolFromSmiles(node_dict['canonical_smiles'])))
        self.testing_df['canonical_smiles'].map(
            lambda x: self.bulk_to_bulk(self.get_prop(x, 'Node'), node, 'Rdkit_Sim_Score',
                                        rdkit_sim_score=self.get_prop(x, 'rdkit_sim_score')))
        self.tx.commit()  # Merge all the relationships created

    """
    Relationship protocol comparing pkas of different molecules
    """

    def compare_pka(self, testing_pka, testing_node, current_pka, current_node):
        if str(current_pka) == 'nan':
            return
        if str(testing_pka) == 'nan':
            return
        current_pka = current_pka.split(',')
        testing_pka = testing_pka.split(',')
        for pka_site in testing_pka:
            if pka_site in current_pka:
                self.bulk_to_bulk(testing_node, current_node, 'similar_pka', pka_site=pka_site)

    def protocol_pka(self, node, node_dict):
        self.tx = self.graph.begin()
        current_pka = node_dict['pka']
        self.testing_df['tf_pka'] = self.testing_df['canonical_smiles'].map(
            lambda x: self.compare_pka(self.get_prop(x, 'pka'), self.get_prop(x, 'Node'),
                                       current_pka, node))
        self.tx.commit()

    """
    All the relationship protocols rely on these function. This is to mass create relationships that can be inserted
    in bulk, and not one by one. 
    """

    def get_prop(self, con_smiles, prop):
        prop_dict = self.testing_df.loc[self.testing_df['canonical_smiles'] == con_smiles].to_dict('records')[0]
        return prop_dict[prop]

    def bulk_to_bulk(self, testing_node, current_node, relationship, **properties):
        Rel = Relationship(testing_node, relationship, current_node, **properties)
        self.tx.create(Rel)  # Create individual relationship

    """
    These methods are the backbone methods that are needed for the main function to work properly 
    """

    def timer(self, time_for_batch):  # Keep track of long it will take to finish
        if not self.o:  # Please refer to excel spreadsheet to understand how this function was derived
            n = self.len_nodes
            self.o = n * (n + 1) / 2
        ni = self.molecules_remaining
        oi = ni * (ni + 1) / 2
        delta_o = self.o - oi
        m = (time_for_batch / delta_o)
        if not self.m:
            self.m = m
        self.m = ((self.counter - 1) * self.m + m) / self.counter
        self.o = oi
        time_needed = self.m * self.o
        return time_needed

    def __get_testing_df__(self, i):  # Get DataFrame to compare molecule to
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
            self.counter = self.counter + 1
            self.molecules_remaining = self.len_nodes - self.counter
            print("{0} molecules left to compare".format(self.molecules_remaining))  # Let user know amount left
            node_dict = (dict(node))
            for i in range(self.splits):
                if (i * self.max_nodes_in_ram + self.max_nodes_in_ram) <= self.counter:
                    pass
                elif i == self.splits - 1 and self.counter >= self.abs_upper_limit:
                    pass
                else:
                    self.testing_df = self.__get_testing_df__(i)
                    self.protocol_dict[self.protocol](node, node_dict)
            time_for_batch = clock() - time_for_batch
            time_left_minutes = round(self.timer(time_for_batch) / 60, 2)
            time_left_hours = round(time_left_minutes / 60, 2)
            self.time_df = self.time_df.append({'Molecules Remaining': self.molecules_remaining,
                                                'Time needed (s)': time_for_batch,
                                                'Total Time passed (min)': self.run_time,
                                                'Predicted Time Left (min)': time_left_minutes}, ignore_index=True)
            print("\nTime Remaining: {0} minutes ({1} hours)".format(time_left_minutes, time_left_hours))
        self.time_df.to_csv('bulkchem_datafiles/Time_vs_molecules.csv', index=False)

    def __init__(self, protocol, max_nodes_in_ram=3000):

        self.run_time = clock()  # Declare variables for timer
        self.average_time = None
        self.o = None
        self.m = None
        self.time_df = pd.DataFrame(columns=['Molecules Remaining', 'Time needed (s)', 'Total Time passed (min)',
                                             'Predicted Time Left (min)'])

        # Please note the timer is still under development and testing, time is likely not accurate

        self.graph = Graph()  # Get neo4j graph database
        self.tx = None

        self.protocol = protocol  # Define protocols
        self.protocol_dict = {
            "Rdkit_sim_score": self.protocol_Rdkit_sim_score,
            "pka": self.protocol_pka
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        matcher = NodeMatcher(self.graph)  # Get nodes from neo4j database (Doesn't load them in ram yet)
        self.raw_nodes = matcher.match("bulkChemMolecule")

        self.len_nodes = len(self.raw_nodes)

        if self.len_nodes <= 0:
            raise Exception("There are no nodes in the database")

        if max_nodes_in_ram > self.len_nodes:  # Verify max nodes does not exceed number of nodes in database
            self.max_nodes_in_ram = self.len_nodes
        else:
            self.max_nodes_in_ram = max_nodes_in_ram

        self.splits = ceil(self.len_nodes / self.max_nodes_in_ram)  # Split data in neo4j database into chunks
        self.abs_upper_limit = len(self.raw_nodes) - (self.splits - 1) * self.max_nodes_in_ram
        self.counter = 0

        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))  # Start comparing
        self.__main__()
