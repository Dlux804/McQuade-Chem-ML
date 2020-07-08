"""
Objectvie: Create molecular fragments, import them into Neo4j and add them into our ontology
"""

import os
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit import Chem
from py2neo import Graph
from tqdm import tqdm

# Connect to Neo4j Destop.
g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def self_smiles(self):
    """
    Get SMILES directly from pipeline
    :param self:
    :return:
    """
    smiles_list = list(self.data['smiles'])
    return smiles_list


def canonical_smiles(smiles_list):
    """
    Create list of canonical SMILES
    :param smiles_list:
    :return:
    """
    canonical_smiles = list(map(Chem.MolToSmiles, list(map(Chem.MolFromSmiles, smiles_list))))  # SMILES to Canonical
    return canonical_smiles


def cypher_smiles_func_groups(smiles, fragment_list):
    """

    :param smiles:
    :param fragment_list:
    :return:
    """
    for fragment in tqdm(fragment_list, desc="Importing fragments to Neo4j"):
        g.evaluate("merge (frag:Fragment {fragment: $fragment})", parameters={'fragment': fragment})
        g.evaluate("match (smiles:SMILES {SMILES:$mol}), (frag:Fragment {fragment: $fragment})"
                   "merge (frag)<-[:HAS_FRAGMENT]-(smiles)",
                   parameters={'fragment': fragment, 'mol': smiles})


def fragments_to_neo(canonical_smiles):
    """"""
    print("Making molecular fragments")

    for smiles in tqdm(canonical_smiles, desc="Creating molecular fragments for SMILES"):
        fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
        fparams = FragmentCatalog.FragCatParams(0, 3, fName)  # I need more research and tuning on this one
        fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
        fcgen = FragmentCatalog.FragCatGenerator()
        mol = Chem.MolFromSmiles(smiles)
        fcount = fcgen.AddFragsFromMol(mol, fcat)
        fcgen.AddFragsFromMol(mol, fcat)
        frag_list = []
        for frag in range(fcount):
            frag_list.append(fcat.GetEntryDescription(frag))  # List of molecular fragments
        cypher_smiles_func_groups(smiles, frag_list)

