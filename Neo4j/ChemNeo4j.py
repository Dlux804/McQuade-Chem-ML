import pandas as pd
from rdkit import Chem, DataStructs
from py2neo import Graph, Node, Relationship
from Function_Group_Search_v3 import Search_Fragments
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

def generate_search_query(label, index, index_value):
    query = r'''match (n:{0} {1}{2}:'{3}'{4}) 
                            RETURN n'''.format(label, '{', index, index_value, '}')
    if '\\' in query:
        query = query.replace("\\", "\\"+"\\")
    return query

def add_con_smiles(file):
    data = pd.read_csv(file)
    smiles = data['Smiles']
    con_smiles = []
    for smile in smiles:
        con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        con_smiles.append(con_smile)
    data['Canonical-Smiles'] = con_smiles
    data.to_csv(file, index=False)

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


class create_relationships:

    def compare_rdkit_score(self, testing_mol, current_mol, min_score):
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        if sim_score >= min_score:
            return True
        else:
            return False

    def compare_molwt(self, testing_mol, current_mol, weight_difference): # Will compare molecular weight
        current_molwt = Chem.Descriptors.ExactMolWt(current_mol)
        testing_molwt = Chem.Descriptors.ExactMolWt(testing_mol)
        current_molwt_lower_bound = current_molwt - current_molwt * weight_difference
        current_molwt_higher_bound = current_molwt + current_molwt * weight_difference
        if current_molwt_higher_bound >= testing_molwt >= current_molwt_lower_bound:
            return True
        else:
            return False

    def compare_fragments(self, testing_mol, current_mol, num_of_matching_frags): # Will compare fragments
        testing_fragments = Search_Fragments(Chem.MolToSmiles(testing_mol)).results
        current_fragments = Search_Fragments(Chem.MolToSmiles(current_mol)).results
        num_of_frags = 0
        for frag in testing_fragments:
            if frag in current_fragments:
                num_of_frags += 1
            if num_of_frags == num_of_matching_frags:
                return True
        return False

    def compare_number_of_carbons(self, testing_mol, current_mol): # Compare carbons
        num_of_carbons_in_testing = len(testing_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        num_of_carbons_in_current = len(current_mol.GetSubstructMatches(Chem.MolFromSmiles('C')))
        if num_of_carbons_in_current == num_of_carbons_in_testing:
            return True
        else:
            return False

    def insert_relationship(self, testing_mol, current_mol, label, relationship):
        graph = self.graph
        testing_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(testing_mol))
        testing_node = graph.evaluate(testing_query)  # Fetch node
        current_query = generate_search_query(label, 'canonical_smiles', Chem.MolToSmiles(current_mol))
        current_node = graph.evaluate(current_query)
        MolWt_Same_Num_Of_Carbons = Relationship(testing_node, relationship, current_node)
        graph.merge(MolWt_Same_Num_Of_Carbons)

    def protocol_A(self, testing_mol, current_mol):
        if self.compare_molwt(testing_mol, current_mol, 0.01):  # Check if molecules have similar molwt
            if self.compare_fragments(testing_mol, current_mol, 2): # Compare number of carbons
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

    def compare_molecules(self): #The main loop, comparing all molecules to each other
        protocol = self.protocol
        for i in range(len(self.bulk_dicts)):
            print("{0} molecules left to compare".format(str(len(self.bulk_dicts)-i))) # Let user know time left
            current_molecule = self.bulk_dicts[i]
            current_mol = Chem.MolFromSmiles(current_molecule['Canonical-Smiles'])
            for x in range(i + 1, len(self.bulk_dicts)):
                testing_molecule = self.bulk_dicts[x]
                testing_mol = Chem.MolFromSmiles(testing_molecule['Canonical-Smiles'])
                if protocol == "A":
                    self.protocol_A(testing_mol, current_mol)
                if protocol == "B":
                    self.protocol_B(testing_mol, current_mol)
                if protocol == "C":
                    self.protocol_C(testing_mol, current_mol)

    def __init__(self, protocol, file=None, retrieve_data_from_neo4j=False):

        # TODO allow to take data from neo4j

        if file is None and retrieve_data_from_neo4j is False:
            raise Exception('Must either give a file name or state "retrieve_data_from_neo4j=True"')

        if retrieve_data_from_neo4j is True:
            raise Exception('Neo4j retrieval not supported yet')

        protocols = ['A', 'B', 'C']
        if protocol not in protocols:
            raise Exception('Protocol not found, try using: "A", "B", or "C"')

        print('Comparing Bulk Chem Data with relationship {0}'.format(protocol))

        self.file = file
        self.protocol = protocol
        self.bulk_data = pd.read_csv(file)
        self.bulk_dicts = self.bulk_data.to_dict('records')
        self.graph = Graph()
        self.compare_molecules()


print("Inserting Bulk Chem Molecules")
insert_bulk_chem_molecules('BulkChemData.csv')
create_relationships('C', file='BulkChemData.csv')