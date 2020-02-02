import pandas as pd
from rdkit import Chem, DataStructs
from py2neo import Graph, Node, Relationship, NodeMatcher
from core.Function_Group_Search_v3 import Search_Fragments  # Import function to search for fragments
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


def generate_search_query(label, index, index_value):  # Weird bug with smiles, this function fixes the bug
    query = r'''match (n:{0} {1}{2}:'{3}'{4}) 
                            RETURN n'''.format(label, '{', index, index_value, '}')
    if '\\' in query:
        query = query.replace("\\", "\\" + "\\")
    return query


def add_con_smiles(file):  # Generate a canonical smiles column if one does not already exist
    data = pd.read_csv(file)
    smiles = data['Smiles']
    con_smiles = []
    for smile in smiles:
        con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        con_smiles.append(con_smile)
    data['Canonical-Smiles'] = con_smiles
    data.to_csv(file, index=False)


def insert_bulk_chem_molecules(file): # Insert the bulk chem molecules into neo4j
    print("Inserting Bulk Chem Molecules")
    graph = Graph()
    bulk_data = pd.read_csv(file)
    bulk_dicts = bulk_data.to_dict('records')
    for i in range(len(bulk_dicts)):
        bulk_dict = bulk_dicts[i]
        bulkChemMolecule = Node("bulkChemMolecule", chemical_name=bulk_dict['Product'], cas=bulk_dict['CAS'],
                                smiles=bulk_dict['Smiles'],
                                canonical_smiles=bulk_dict['Canonical-Smiles'], fragements=bulk_dict['Fragments'],
                                num_of_carbons=str(bulk_dict['Num of Carbons']),
                                chiral_center=bulk_dict['Chiral Center'],
                                molecular_weight=str(bulk_dict['Molecular Weight (g/mol)']))
        graph.merge(bulkChemMolecule, 'bulkChemMolecule', 'chemical_name')


class create_relationships:  # Class to generate the different relationship protocols

    @staticmethod
    def compare_rdkit_score(testing_mol, current_mol, min_score):  # Compare rdkit score
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        if sim_score >= min_score:
            return True
        else:
            return False

    @staticmethod
    def compare_molwt(testing_mol, current_mol, weight_difference):  # Will compare molecular weight
        current_molwt = Chem.Descriptors.ExactMolWt(current_mol)
        testing_molwt = Chem.Descriptors.ExactMolWt(testing_mol)
        current_molwt_lower_bound = current_molwt - current_molwt * weight_difference
        current_molwt_higher_bound = current_molwt + current_molwt * weight_difference
        if current_molwt_higher_bound >= testing_molwt >= current_molwt_lower_bound:
            return True
        else:
            return False

    @staticmethod
    def compare_fragments(testing_mol, current_mol, num_of_matching_frags):  # Will compare fragments
        testing_fragments = Search_Fragments(Chem.MolToSmiles(testing_mol)).results
        current_fragments = Search_Fragments(Chem.MolToSmiles(current_mol)).results
        num_of_frags = 0
        for frag in testing_fragments:
            if frag in current_fragments:
                num_of_frags += 1
            if num_of_frags == num_of_matching_frags:
                return True
        return False

    @staticmethod
    def compare_number_of_carbons(testing_mol, current_mol):  # Compare carbons
        num_of_carbons_in_testing = len(testing_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        num_of_carbons_in_current = len(current_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        if num_of_carbons_in_current == num_of_carbons_in_testing:
            return True
        else:
            return False

    def insert_relationship(self, testing_mol, current_mol, label, relationship):  # General function to insert relationships
        graph = self.graph
        testing_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(testing_mol))
        testing_node = graph.evaluate(testing_query)  # Fetch node
        current_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(current_mol))
        current_node = graph.evaluate(current_query)
        MolWt_Same_Num_Of_Carbons = Relationship(testing_node, relationship, current_node)
        graph.merge(MolWt_Same_Num_Of_Carbons)

    def protocol_A(self, testing_mol, current_mol):
        if self.compare_molwt(testing_mol, current_mol, 0.01):  # Check if molecules have similar molwt
            if self.compare_fragments(testing_mol, current_mol, 2):  # Compare number of carbons
                self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Molwt_Carbons')
            if self.compare_number_of_carbons(testing_mol, current_mol):  # Compare number of matching fragments
                self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Molwt_Frags')

    def protocol_B(self, testing_mol, current_mol):
        if self.compare_rdkit_score(testing_mol, current_mol, 0.95):
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Sim')

    def protocol_C(self, testing_mol, current_mol):
        if self.compare_molwt(testing_mol, current_mol, 0.01):
            if self.compare_fragments(testing_mol, current_mol, 2):
                if self.compare_rdkit_score(testing_mol, current_mol, 0.95):
                    self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Molwt_Sim_Frags')

    def protocol_D(self, testing_mol, current_mol):
        if self.compare_molwt(testing_mol, current_mol, 0.05):
            if self.compare_rdkit_score(testing_mol, current_mol, 0.9):
                self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Molwt_Sim')

    def protocol_E(self, testing_mol, current_mol):
        result_dict = {'Molwt': (self.compare_molwt(testing_mol, current_mol, 0.05)),
                       'Rdkit': (self.compare_rdkit_score(testing_mol, current_mol, 0.95)),
                       'Frags': (self.compare_fragments(testing_mol, current_mol, 2)),
                       'NumCarbons': (self.compare_number_of_carbons(testing_mol, current_mol))}
        results_list = []
        for key in result_dict.keys():
            if result_dict[key]:
                results_list.append(key)
        if results_list:
            relationship = '_'.join(results_list)
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', relationship)

    def protocol_F(self, testing_mol, current_mol):  # This is by far the best protocol, please use this
        if self.compare_molwt(testing_mol, current_mol, 0.05):
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Similar_Molecular_Weight')
        if self.compare_rdkit_score(testing_mol, current_mol, 0.95):
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'High_Rdkit_Sim_Score')
        if self.compare_fragments(testing_mol, current_mol, 2):
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Matching_Functional_Groups')
        if self.compare_number_of_carbons(testing_mol, current_mol):
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Same_Number_Of_Carbons')

    def compare_molecules(self):  # The main loop, comparing all molecules to each other
        for i in range(len(self.bulk_dicts)):
            print("{0} molecules left to compare".format(str(len(self.bulk_dicts) - i)))  # Let user know amount left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            for x in range(i + 1, len(self.bulk_dicts)):
                testing_molecule = self.bulk_dicts[x]
                testing_mol = Chem.MolFromSmiles(testing_molecule['Canonical-Smiles'])
                self.protocol_dict[self.protocol](testing_mol, current_mol)

    def __init__(self, protocol, file=None, retrieve_data_from_neo4j=False):

        self.graph = Graph()  # Get neo4j graph database

        self.protocol = protocol  # Define protocol
        self.protocol_dict = {
            "A": self.protocol_A,
            "B": self.protocol_B,
            "C": self.protocol_C,
            "D": self.protocol_D,
            "E": self.protocol_E,
            "F": self.protocol_F
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        if file is None and retrieve_data_from_neo4j is False:  # Make sure user pointed to where data is
            raise Exception('Must either give a file name or state "retrieve_data_from_neo4j=True"')

        if retrieve_data_from_neo4j:  # Get the data from neo4j
            matcher = NodeMatcher(self.graph)
            raw_nodes = matcher.match("bulkChemMolecule")
            bulk_dicts = []
            for node in raw_nodes:
                node = (dict(node))
                node['Canonical-Smiles'] = node['canonical_smiles']
                del node['canonical_smiles']
                bulk_dicts.append(node)
            self.bulk_dicts = bulk_dicts

        if file is not None:  # Get data from bulkchem csv
            bulk_data = pd.read_csv(file)
            self.bulk_dicts = bulk_data.to_dict('records')

        print('Comparing Bulk Chem Data with relationship {0}'.format(self.protocol))
        self.compare_molecules()


insert_bulk_chem_molecules('BulkChemData.csv')
create_relationships('F', retrieve_data_from_neo4j=True)
