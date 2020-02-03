import pandas as pd
from rdkit import Chem, DataStructs
from py2neo import Graph, Node, Relationship, NodeMatcher
from core.fragments import Search_Fragments  # Import function to search for fragments
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

    def compare_rdkit_score(self, testing_mol, current_mol, min_score):  # Compare rdkit score
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        if sim_score >= min_score:
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'High_Rdkit_Sim_Score')

    def compare_molwt(self, testing_mol, current_mol, weight_difference):  # Will compare molecular weight
        current_molwt = Chem.Descriptors.ExactMolWt(current_mol)
        testing_molwt = Chem.Descriptors.ExactMolWt(testing_mol)
        current_molwt_lower_bound = current_molwt - current_molwt * weight_difference
        current_molwt_higher_bound = current_molwt + current_molwt * weight_difference
        if current_molwt_higher_bound >= testing_molwt >= current_molwt_lower_bound:
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Similar_Molecular_Weight')

    def compare_fragments(self, testing_mol, current_mol, num_of_matching_frags):  # Will compare fragments
        if num_of_matching_frags <= 0:
            raise Exception('The Number of matching fragments must be greater than 0')
        testing_fragments = Search_Fragments(Chem.MolToSmiles(testing_mol)).results
        current_fragments = Search_Fragments(Chem.MolToSmiles(current_mol)).results
        num_of_frags = 0
        for frag in testing_fragments:
            if frag in current_fragments:
                num_of_frags += 1
            if num_of_frags == num_of_matching_frags:
                self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Matching_Functional_Groups')

    def compare_number_of_carbons(self, testing_mol, current_mol):  # Compare carbons
        num_of_carbons_in_testing = len(testing_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        num_of_carbons_in_current = len(current_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        if num_of_carbons_in_current == num_of_carbons_in_testing:
            self.insert_relationship(testing_mol, current_mol, 'bulkChemMolecule', 'Same_Number_Of_Carbons')

    def insert_relationship(self, testing_mol, current_mol, label, relationship):  # General function to insert relationships
        testing_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(testing_mol))
        testing_node = self.graph.evaluate(testing_query)  # Fetch node
        current_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(current_mol))
        current_node = self.graph.evaluate(current_query)
        Rel = Relationship(testing_node, relationship, current_node)  # Gen Relationship
        self.graph.merge(Rel)  # Merge relationship

    def protocol_A(self, testing_df, current_mol):
        """
        This is a more optimized code of the protocol_F. Instead of going through and comparing each molecule one by
        one. This script will compare one molecule to all of the remaining molecules all at once for each the different
        functions to define relationships. This is a 100x speed up on my laptop compared to the previous code.
        """
        testing_df['Canonical-Smiles'].map(
            lambda x: self.compare_rdkit_score(Chem.MolFromSmiles(x), current_mol, 0.95))
        testing_df['Canonical-Smiles'].map(
            lambda x: self.compare_molwt(Chem.MolFromSmiles(x), current_mol, 0.05))
        testing_df['Canonical-Smiles'].map(
            lambda x: self.compare_fragments(Chem.MolFromSmiles(x), current_mol, 2))
        testing_df['Canonical-Smiles'].map(
            lambda x: self.compare_number_of_carbons(Chem.MolFromSmiles(x), current_mol, 0.95))

    def compare_molecules(self):  # The main loop, comparing all molecules to each other
        for i in range(len(self.bulk_dicts)):
            print("{0} molecules left to compare".format(str(len(self.bulk_dicts) - i)))  # Let user know amount left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            testing_df = self.bulk_data[i+1:]
            self.protocol_A(testing_df, current_mol)

    def __init__(self, protocol, file=None, retrieve_data_from_neo4j=False):

        self.graph = Graph()  # Get neo4j graph database

        self.protocol = protocol  # Define protocol
        self.protocol_dict = {
            "A" : self.protocol_A
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
            self.bulk_data = pd.DataFrame(bulk_dicts)

        if file is not None:  # Get data from bulkchem csv
            bulk_data = pd.read_csv(file)
            self.bulk_dicts = bulk_data.to_dict('records')

        print('Comparing Bulk Chem Data with relationship {0}'.format(self.protocol))
        self.compare_molecules()


insert_bulk_chem_molecules('BulkChemData.csv')
create_relationships('A', retrieve_data_from_neo4j=True)
