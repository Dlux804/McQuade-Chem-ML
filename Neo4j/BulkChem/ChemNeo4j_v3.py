from time import clock
from py2neo import Graph, Relationship, NodeMatcher
from rdkit import Chem, DataStructs
import pandas as pd
pd.options.mode.chained_assignment = None


class create_relationships:  # Class to generate the different relationship protocols

    def compare_rdkit_score(self, testing_smiles, current_mol):  # Compare rdkit score
        testing_mol = Chem.MolFromSmiles(testing_smiles)
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        return sim_score

    def bulk_to_bulk(self, testing_df, testing_smiles, current_node, label, relationship):
        testing_molecule = testing_df.loc[testing_df['Canonical-Smiles'] == testing_smiles].to_dict('records')[0]
        Rel = Relationship(testing_molecule['Node'], relationship, current_node,
                           rdkit_sim_score=testing_molecule['rdkit_sim_score'])
        self.tx.create(Rel)  # Merge relationship

    def protocol_Default(self, testing_df, current_mol, current_node):
        self.tx = self.graph.begin()
        testing_df['rdkit_sim_score'] = testing_df['Canonical-Smiles'].map(
            lambda x: self.compare_rdkit_score(x, current_mol))
        testing_df['Canonical-Smiles'].map(
            lambda x: self.bulk_to_bulk(testing_df, x, current_node,'bulkChemMolecule', 'Rdkit_Sim_Score'))
        self.tx.commit()

    def compare_molecules(self):  # The main loop, comparing all molecules to each other
        time_df = pd.DataFrame(columns=['Molecules Remaining', 'Time needed (s)', 'Total Time passed (min)'])
        total_time_passed = clock()
        for i in range(len(self.bulk_dicts)):
            time_needed = clock()
            molecules_remaining = str(len(self.bulk_dicts) - i)
            print("{0} molecules left to compare".format(molecules_remaining))  # Let user know amount left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            current_node = current_molecule['Node']
            testing_df = self.bulk_data[i + 1:]
            self.protocol_dict[self.protocol](testing_df, current_mol, current_node)
            total_time_passed = (clock() - total_time_passed) / 60
            time_needed = clock() - time_needed
            time_df = time_df.append({'Molecules Remaining': molecules_remaining, 'Time needed (s)': time_needed,
                                     'Total Time passed (min)': total_time_passed}, ignore_index=True)
            print("{0} seconds needed for batch".format(round(time_needed, 2)))
            print("{0} minutes have passed\n".format(round(total_time_passed, 2)))
            time_df.to_csv('Time_vs_molecules.csv', index=False)

    def __init__(self, protocol, file=None, retrieve_data_from_neo4j=False):

        self.graph = Graph()  # Get neo4j graph database

        self.protocol = protocol  # Define protocol
        self.protocol_dict = {
            "Default": self.protocol_Default,
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        matcher = NodeMatcher(self.graph)
        self.raw_nodes = matcher.match("bulkChemMolecule")
        bulk_dicts = []
        for node in self.raw_nodes:
            molecule = (dict(node))
            molecule['Canonical-Smiles'] = molecule['canonical_smiles']
            del molecule['canonical_smiles']
            molecule['Node'] = node
            bulk_dicts.append(molecule)
        self.bulk_dicts = bulk_dicts
        self.bulk_data = pd.DataFrame(bulk_dicts)

        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))
        self.compare_molecules()