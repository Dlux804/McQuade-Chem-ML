import pandas as pd
from rdkit import Chem, DataStructs
from py2neo import Graph, Node, Relationship, NodeMatcher
from Function_Group_Search_v3 import Search_Fragments
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


#  This function will insert the different fragments from 'List_of_fragments_groups.txt' as different nodes
def insert_fragments(file):
    with open(file, 'r') as file:
        fragments = file.readline().split(', ')
    graph = Graph()
    i = 0
    for fragment in fragments:
        fragment_node = Node('Fragment', fragment_ID='{0}'.format(str(fragment)),
                             functional_group='{0}'.format(str(fragment)))
        graph.merge(fragment_node, "functional_group", "fragment_ID")
        i = i + 1


#  This function will insert the different molecules in 'BulkChemData.csv'
def insert_bulk_chem_molecules(file):
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


#  This function will add a column for canonical smiles if it does not already exist
def add_con_smiles(file):
    data = pd.read_csv(file)
    smiles = data['Smiles']
    con_smiles = []
    for smile in smiles:
        con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        con_smiles.append(con_smile)
    data['Canonical-Smiles'] = con_smiles
    data.to_csv(file, index=False)


def insert_Kaitlin_stuff(file):
    graph = Graph()
    pass


def generate_rdkit_data(file):
    pass


class create_relationships_B:

    '''
    This is an rdkit comparision. Rdkit has a function to give a score for similarity. The range for the given score
    is 0-1. Lets relate molecules that have a similarity score of 0.95 or higher.
    '''

    def compare_molecules(self): # The main loop, comparing all molecules to one another
        for i in range(len(self.bulk_dicts)):
            print("{0} molecules left to compare".format(str(len(self.bulk_dicts)-i))) # Let user know time left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            for x in range(i + 1, len(self.bulk_dicts)):
                testing_molecule = self.bulk_dicts[x]
                testing_mol = Chem.MolFromSmiles(testing_molecule['Canonical-Smiles'])
                self.compare_rdkit_score(testing_mol, current_mol)

    def compare_rdkit_score(self, testing_mol, current_mol):
        graph = self.graph
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        if sim_score >= 0.95:
            testing_query = "match (n:bulkChemMolecule {canonical_smiles:'"+\
                            Chem.MolToSmiles(testing_mol)+"'}) RETURN n"
            testing_node = graph.evaluate(testing_query)  # Fetch node
            current_query = "match (n:bulkChemMolecule {canonical_smiles:'" + \
                            Chem.MolToSmiles(current_mol) + "'}) RETURN n"
            current_node = graph.evaluate(current_query)
            Rdkit_sim_score = Relationship(testing_node, "Rdkit_sim_score", current_node)  # Create rel
            graph.merge(Rdkit_sim_score)  # Merge relationship

    def __init__(self, file):
        self.file = file
        self.bulk_data = pd.read_csv(file)
        self.bulk_dicts = self.bulk_data.to_dict('records')
        self.graph = Graph()
        self.compare_molecules()


class create_relationships_A:

    """
    This is a trivial comparison with the bulk chem data. This will compare the molecular weight between
    two molecules. If their weight is within 1% of each other, then continue matching. If a molecule either has
    the same number of carbon, or two fragments, then relate them.
    """

    # TODO change to retrive data from neo4j, rather than take it from the bulkchemdata
    def compare_molecules(self): #The main loop, comparing all molecules to each other
        for i in range(len(self.bulk_dicts)):
            print("{0} molecules left to compare".format(str(len(self.bulk_dicts)-i))) # Let user know time left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            for x in range(i + 1, len(self.bulk_dicts)):
                testing_molecule = self.bulk_dicts[x]
                testing_mol = Chem.MolFromSmiles(testing_molecule['Canonical-Smiles'])
                if self.compare_molwt(testing_mol, current_mol): # Check if molecules have similar molwt
                    self.compare_fragments(testing_mol, current_mol) # Compare number of carbons
                    self.compare_number_of_carbons(testing_mol, current_mol) # Compare number of matching fragments

    def compare_molwt(self, testing_mol, current_mol): # Will compare molecular weight
        graph = self.graph
        current_molwt = Chem.Descriptors.ExactMolWt(current_mol)
        testing_molwt = Chem.Descriptors.ExactMolWt(testing_mol)
        current_molwt_lower_bound = current_molwt - current_molwt * 0.01
        current_molwt_higher_bound = current_molwt + current_molwt * 0.01
        if current_molwt_higher_bound >= testing_molwt >= current_molwt_lower_bound:
            return True
        else:
            return False

    def compare_fragments(self, testing_mol, current_mol): # Will compare fragments
        graph = self.graph
        testing_fragments = Search_Fragments(Chem.MolToSmiles(testing_mol)).results
        current_fragments = Search_Fragments(Chem.MolToSmiles(current_mol)).results
        num_of_frags = 0
        for frag in testing_fragments:
            if frag in current_fragments and frag != 'Arene':
                num_of_frags += 1
            if num_of_frags == 2:
                testing_query = "match (n:bulkChemMolecule {canonical_smiles:'" + \
                                Chem.MolToSmiles(testing_mol) + "'}) RETURN n" # Cypher query to retrive node
                testing_node = graph.evaluate(testing_query)  # Fetch node
                current_query = "match (n:bulkChemMolecule {canonical_smiles:'" + \
                                Chem.MolToSmiles(current_mol) + "'}) RETURN n"
                current_node = graph.evaluate(current_query)
                MolWt_HAS_FRAGMENT = Relationship(testing_node, "MolWt_HAS_FRAGMENT", current_node) # Create rel
                graph.merge(MolWt_HAS_FRAGMENT) # Merge relationship
                break # End loop, since relationship already established

    def compare_number_of_carbons(self, testing_mol, current_mol): # Compare carbons
        graph = self.graph
        num_of_carbons_in_testing = len(testing_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        num_of_carbons_in_current = len(current_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        if num_of_carbons_in_current == num_of_carbons_in_testing:
            testing_query = "match (n:bulkChemMolecule {canonical_smiles:'" + \
                            Chem.MolToSmiles(testing_mol) + "'}) RETURN n"
            testing_node = graph.evaluate(testing_query)  # Fetch node
            current_query = "match (n:bulkChemMolecule {canonical_smiles:'" + \
                            Chem.MolToSmiles(current_mol) + "'}) RETURN n"
            current_node = graph.evaluate(current_query)
            MolWt_Same_Num_Of_Carbons = Relationship(testing_node, "MolWt_Same_Num_Of_Carbons", current_node)
            graph.merge(MolWt_Same_Num_Of_Carbons)

    def __init__(self, file):
        self.file = file
        self.bulk_data = pd.read_csv(file)
        self.bulk_dicts = self.bulk_data.to_dict('records')
        self.graph = Graph()
        self.compare_molecules()

'''
Before running this script, make sure to assert that canonical smiles are unique, you can use the cypher command

CREATE CONSTRAINT ON (r:bulkChemMolecule)
ASSERT r.canonical_smiles IS UNIQUE
'''

print('Inserting Bulk Chem Data') # Insert the bulk chem molecules
insert_bulk_chem_molecules("BulkChemData.csv")

# print('Comparing Bulk Chem Data with relationship A') # Compare using relationship A
# create_relationships_A('BulkChemData.csv')

print('Comparing Bulk Chem Data with relationship B') # Compare using relationship B
create_relationships_B('BulkChemData.csv')

