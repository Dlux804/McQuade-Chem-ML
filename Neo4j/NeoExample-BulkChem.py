import pandas as pd
from rdkit import Chem
from py2neo import Graph, Node, Relationship

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


def insert_bulk_chem_molecules(file):
    graph = Graph()
    bulk_data = pd.read_csv(file)
    columns = list(bulk_data.columns)
    for i in range(len(bulk_data)):
        line = list(bulk_data.iloc[i, :])
        dict = {}
        for i in range(len(line)):
            if i == 6 and str(line[6]) != 'nan':
                fragments = line[6].split(', ')
            else:
                dict[columns[i]] = line[i]

        bulkChemMolecule = Node("bulkChemMolecule", cas=dict['CAS'], chemical_name=dict['Product'], smiles=dict['Smiles'],
                                canonical_smiles=dict['Canonical-Smiles'],
                                num_of_carbons=str(dict['Num of Carbons']), chiral_center=dict['Chiral Center'],
                                molecular_weight=str(dict['Molecular Weight (g/mol)']))

        graph.merge(bulkChemMolecule, 'chemical_name', 'cas')

        for fragment in fragments:
            fragment_node = graph.nodes.match("Fragment", fragment_ID=fragment).first()
            HAS_FRAGMENT = Relationship.type("HAS_FRAGMENT")
            graph.merge(HAS_FRAGMENT(bulkChemMolecule, fragment_node))


def add_con_smiles(file):
    data = pd.read_csv(file)
    smiles = data['Smiles']
    con_smiles = []
    for smile in smiles:
        con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        con_smiles.append(con_smile)
    data['Canonical-Smiles'] = con_smiles
    data.to_csv(file)


def insert_Kaitlin_stuff(file):
    pass


#insert_fragments('List_of_fragment_groups.txt')
#insert_bulk_chem_molecules("BulkChemData.csv")
add_con_smiles('BulkChemData.csv')


