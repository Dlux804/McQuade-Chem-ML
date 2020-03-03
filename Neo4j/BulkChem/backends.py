import pandas as pd
from rdkit import Chem
from py2neo import Node, Relationship
from io import StringIO
import numpy as np

'''
The point of this file to initialize the Neo4j graph database. This script is designed mainly for the bulkchemdata, but
will at some in the future add to allow for other data sources (such as pubchem, NIST, etc.)
'''


def generate_search_query(label, index, index_value):  # Weird bug with smiles, this function fixes the bug
    query = r'''match (n:{0} {1}{2}:'{3}'{4}) RETURN n'''.format(label, '{', index, index_value, '}')
    if '\\' in query:
        query = query.replace("\\", "\\" + "\\")
    return query


def get_fragments(smiles):
    smiles_mol = Chem.MolFromSmiles(smiles)
    if smiles_mol is None:
        return None

    fragments_df = pd.read_csv('bulkchem_datafiles/Function-Groups-SMARTS.csv')
    frags = []
    for i in range(0, len(fragments_df)):
        row = dict(fragments_df.loc[i, :])
        if str(row['Sub-SMARTS']) != 'nan':
            sub_mol = Chem.MolFromSmarts(str(row['Sub-SMARTS']))
            main_mol = Chem.MolFromSmarts(str(row['SMARTS']))
            sub_mol_matches = len(smiles_mol.GetSubstructMatches(sub_mol))
            main_mol_matches = len(smiles_mol.GetSubstructMatches(main_mol))
            if sub_mol_matches == 0:
                pass
            elif sub_mol_matches == main_mol_matches:
                frags.append(row['Fragment'])
            elif sub_mol_matches > main_mol_matches > 0:
                frags.append(row['Fragment'])
                frags.append(row['Sub-Fragment'])
            else:
                frags.append(row['Sub-Fragment'])
        else:
            main_mol = Chem.MolFromSmarts(str(row['SMARTS']))
            main_mol_matches = len(smiles_mol.GetSubstructMatch(main_mol))
            if main_mol_matches > 0:
                frags.append(row['Fragment'])
    frags = list(set(frags))
    return ", ".join(frags)

def get_prop_from_smiles_df(df, con_smiles, prop):
    prop_dict = df.loc[df['canonical_smiles'] == con_smiles].to_dict('records')[0]
    return prop_dict[prop]


"""
These methods are the backbone methods that are needed for the main function to work properly.

The timer keeps track of the time needed until all entities are related and inserted. Instead of explaining how
the math works in words in this file, please refer to 'bulkchem_datafiles/Time_vs_molecules_demo.xlsx'. The file
can serve as a visual explanation to understand the math behind this function. But in summary, the time needed to 
trained is directly related to number of operations remaining. Time Remaining = m * Operations left:
where m = change in time/change in operations. The m is a changing average that is calculated as the relationships
are inserted. This function basically calculates the moving average m, then multiples that by the number of predict
operations left.
    Now, the actual number of operations will vary in reality. Since it is not possible to know exactly how many 
    total operations will be needed, as we do not know how many relationships will be inserted into the GB. So 
    this timer only acts as an estimate, and will not have perfect predicating power. But, it is assumed that
    all molecules will have a relationship, so the predict acts as a upper limit for the max amount of time the
    script will need to run based on the moving average.

The '__get_testing_df__' function is responsible for loading in a set number of nodes at a time. For bulkchem
database, there are a small number of molecules (nodes) and all of them can be loaded into memory safely. But, for
larger databases, such as PubChem, this is not a good idea. So this functions will split the nodes in the Neo4j
GB into chunks that can be digested and inserted in bulk safely. So if there were 300 total nodes, and each node
was to be compared with each other node and the max nodes to keep in memory is 100, then there would be a max of 3
chunks to digest. I say max because if more than 100 nodes have already been compared, then there are only
200 nodes left to compare. And since there 100 max nodes allowed to be keep in memory, this difference has to be
keep track of. This difference is easiest to be keep track of in the main function. 
    So this function keeps track of the number of node remaining and the max number of nodes to keep in
    memory, and return the testing DataFrame accordingly. 
"""


def get_testing_df(i, raw_nodes, max_nodes_in_ram, counter):
    lower_limit = i * max_nodes_in_ram + counter
    upper_limit = lower_limit + max_nodes_in_ram
    bulk_dicts = []
    counter = lower_limit
    for node in raw_nodes.skip(lower_limit):
        if counter == upper_limit:
            break
        molecule = (dict(node))
        molecule['Node'] = node
        bulk_dicts.append(molecule)
        counter = counter + 1
    return pd.DataFrame(bulk_dicts)


def __timer__(o2, m2, counter, len_nodes, molecules_remaining,
              time_for_batch, time_df, run_time):  # Keep track of long it will take to finish
    if not o2:  # Please refer to excel spreadsheet to understand how this function was derived
        n = len_nodes
        o2 = n * (n + 1) / 2
    n = molecules_remaining
    o1 = n * (n + 1) / 2
    delta_o = o2 - o1
    m1 = (time_for_batch / delta_o)
    if not m2:
        m2 = m1
    m = ((counter - 1) * m1 + m2) / counter
    time_needed = m * o1

    time_left_minutes = round(time_needed / 60, 2)
    time_left_hours = round(time_left_minutes / 60, 2)
    time_df = time_df.append({'Molecules Remaining': molecules_remaining,
                                        'Time needed (s)': time_for_batch,
                                        'Total Time passed (min)': run_time,
                                        'Predicted Time Left (min)': time_left_minutes}, ignore_index=True)
    print("\nTime Remaining: {0} minutes ({1} hours)".format(time_left_minutes, time_left_hours))
    time_df.to_csv('bulkchem_datafiles/Time_vs_molecules.csv', index=False)

    return time_needed, m, o1, time_df


class init_neo_bulkchem:
    """
    The init takes all the data from the bulkchem data, add con smiles if they do not already exist, and insert
    each of the molecule as nodes. The properties of the nodes are the other pieces of information inside of the
    bulkchem data. Also, the user has the option to insert the fragments (or functional groups) as nodes. Whether or
    not the user uses this options depends on the relationships planned for the GB.
    """

    def __init__(self, graph, fragments_as_nodes=True, bulk_chem_data='bulkchem_datafiles/BulkChemData_Orginial.csv'):
        print("Initializing Bulk Chem Data...")
        self.fragments_as_nodes = fragments_as_nodes
        self.graph = graph
        self.bulk_chem_data = pd.read_csv(bulk_chem_data)
        self.__add_con_smiles__()  # Almost all functions in ChemNeo4j rely on con smiles, make sure nodes have them
        self.bulk_dicts = self.bulk_chem_data.to_dict('records')
        if self.fragments_as_nodes:
            self.insert_fragments()
        self.insert_bulk_chem_molecules()
        print("Done")

    def __add_con_smiles__(self):  # Generate a canonical smiles column if one does not already exist
        smiles = self.bulk_chem_data['Smiles']
        con_smiles = []
        for smile in smiles:
            con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
            con_smiles.append(con_smile)
        self.bulk_chem_data['Canonical-Smiles'] = con_smiles

    def insert_bulk_chem_molecules(self):  # Insert the bulk chem molecules into neo4j
        for molecule in self.bulk_dicts:
            bulkChemMolecule = Node("bulkChemMolecule", chemical_name=molecule['Product'], cas=molecule['CAS'],
                                    smiles=molecule['Smiles'],
                                    canonical_smiles=molecule['Canonical-Smiles'], fragments=molecule['Fragments'],
                                    num_of_carbons=str(molecule['Num of Carbons']),
                                    chiral_center=molecule['Chiral Center'],
                                    molecular_weight=str(molecule['Molecular Weight (g/mol)']),
                                    pka=str(molecule['pka']))
            self.graph.merge(bulkChemMolecule, 'bulkChemMolecule', 'chemical_name')
            if self.fragments_as_nodes and str(molecule['Fragments']) != 'nan':
                mol = molecule['Canonical-Smiles']
                for fragment in molecule['Fragments'].split(', '):
                    self.rel_bulkChem_to_fragment(mol, fragment)

    def insert_fragments(self):

        """
        The fragments csv is saved directly into this script to save space and memory. And the string is read as if
        it is a file.
        """

        fragment_data = pd.read_csv('bulkchem_datafiles/Function-Groups-SMARTS.csv')
        fragment_dicts = fragment_data.to_dict('records')
        for fragment_dict in fragment_dicts:
            fragment_data = []
            for thing in fragment_dict:
                if str(fragment_dict[thing]) != 'nan':
                    fragment_data.append(fragment_dict[thing])
            if fragment_data:
                fragmentMolecule = Node("fragmentMolecule", fragment_name=fragment_data[0], smarts=fragment_data[1])
                self.graph.merge(fragmentMolecule, "fragmentMolecule", 'fragment_name')
                if len(fragment_data) > 2:
                    fragmentMolecule = Node("fragmentMolecule", fragment_name=fragment_data[2], smarts=fragment_data[3])
                    self.graph.merge(fragmentMolecule, "fragmentMolecule", 'fragment_name')

    def rel_bulkChem_to_fragment(self, mol, fragment):
        mol_query = generate_search_query('bulkChemMolecule', 'canonical_smiles', mol)
        mol_node = self.graph.evaluate(mol_query)  # Fetch node
        fragment_query = generate_search_query('fragmentMolecule', 'fragment_name', fragment)
        fragment_node = self.graph.evaluate(fragment_query)
        Rel = Relationship(mol_node, 'has_fragment', fragment_node)
        self.graph.merge(Rel)