import pandas as pd
from rdkit import Chem, DataStructs
from py2neo import Graph, Node, Relationship, NodeMatcher
from io import StringIO


def generate_search_query(label, index, index_value):  # Weird bug with smiles, this function fixes the bug
    query = r'''match (n:{0} {1}{2}:'{3}'{4}) RETURN n'''.format(label, '{', index, index_value, '}')
    if '\\' in query:
        query = query.replace("\\", "\\" + "\\")
    return query


class init_neo_bulkchem:

    def __init__(self, fragments_as_nodes=True, bulk_chem_data='BulkChemData.csv'):
        print("Initializing Bulk Chem Data...")
        self.fragments_as_nodes = fragments_as_nodes
        self.graph = Graph()
        self.bulk_chem_data = pd.read_csv(bulk_chem_data)
        self.add_con_smiles()
        self.bulk_dicts = self.bulk_chem_data.to_dict('records')
        if self.fragments_as_nodes:
            self.insert_fragments()
        self.insert_bulk_chem_molecules()
        print("Done")

    def add_con_smiles(self):  # Generate a canonical smiles column if one does not already exist
        smiles = self.bulk_chem_data['Smiles']
        con_smiles = []
        for smile in smiles:
            con_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
            con_smiles.append(con_smile)
        self.bulk_chem_data['Canonical-Smiles'] = con_smiles

    def insert_bulk_chem_molecules(self): # Insert the bulk chem molecules into neo4j
        for molecule in self.bulk_dicts:
            bulkChemMolecule = Node("bulkChemMolecule", chemical_name=molecule['Product'], cas=molecule['CAS'],
                                    smiles=molecule['Smiles'],
                                    canonical_smiles=molecule['Canonical-Smiles'], fragements=molecule['Fragments'],
                                    num_of_carbons=str(molecule['Num of Carbons']),
                                    chiral_center=molecule['Chiral Center'],
                                    molecular_weight=str(molecule['Molecular Weight (g/mol)']))
            self.graph.merge(bulkChemMolecule, 'bulkChemMolecule', 'chemical_name')
            if self.fragments_as_nodes and str(molecule['Fragments']) != 'nan':
                mol = molecule['Canonical-Smiles']
                for fragment in molecule['Fragments'].split(', '):
                    self.rel_bulkChem_to_fragment(mol, fragment)

    def insert_fragments(self):

        fragment_csv = StringIO("""Fragment,SMARTS,Sub-Fragment,Sub-SMARTS,
Aldehyde,[CX3H1](=O)[#6],,,
Ester,[#6][CX3](=O)[OX2H0][#6],Ether,[OD2]([#6])[#6],
Amide,[NX3][CX3](=[OX1])[#6],,,
Carbamate,"[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",,,
Amine,"[NX3;H2,H1;!$(NC=O)]",,,
Imine,"[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",,,
Arene,c,,,
Thiol,[#16X2H],,,
Acyl Halide,"[CX3](=[OX1])[F,Cl,Br,I]",Alkyl Halide,"[#6][F,Cl,Br,I]",
Allenic Carbon,[$([CX2](=C)=C)],,,
Vinylic Carbon,[$([CX3]=[CX3])],,,
Ketone,[#6][CX3](=O)[#6],,,
Carboxylic Acid,[CX3](=O)[OX2H1],Alcohol,[OX2H],
Thioamide,[NX3][CX3]=[SX1],,,
Nitrate,"[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",,,
Nitrile,[NX1]#[CX2],,,
Phenol,[OX2H][cX3]:[c],,,
Peroxide,"[OX2,OX1-][OX2,OX1-]",,,
Sulfide,[#16X2H0],,,
Nitro,"[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",,,
Phosphoric Acid,"[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",Phosphoric Ester,"[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"
        """)

        fragment_data = pd.read_csv(fragment_csv)
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