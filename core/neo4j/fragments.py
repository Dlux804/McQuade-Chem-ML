"""
Objectvie: Create molecular fragments, import them into Neo4j and add them into our ontology
"""

import os

from rdkit import RDConfig
from rdkit import Chem
from core.neo4j.make_query import Query
# Connect to Neo4j Destop.
from rdkit.Chem import FragmentCatalog
from py2neo import ClientError

# TODO REDO DOCSTRINGS


def self_smiles(self):
    """
    Objective: Get SMILES directly from pipeline
    Intent: I created this function only when the user is creating Neo4j graphs directly from the ML pipeline
    :param self:
    :return:
    """
    return list(self.data['smiles'])


def canonical_smiles(smiles_list):
    """
    Objective: Create list of canonical SMILES from SMILES
    Intent: While the SMILES in dataset from moleculenet.ai are all canonical, it is always good to be safe. I don't
            know if I should add in a way to detect irregular SMILES and remove the rows that contains them in the
            dataframe. However, that process should be carried out at the start of the pipeline instead of at the end.
    :param smiles_list:
    :return:
    """
    return list(map(Chem.MolToSmiles, list(map(Chem.MolFromSmiles, smiles_list))))  # SMILES to Canonical


def smiles_to_frag(smiles):
    """
    Objective: Create fragments
    Intent: This script is based on Adam's "mol_frag.ipynb" file in his deepml branch, which is based on rdkit's
            https://www.rdkit.org/docs/GettingStartedInPython.html. I still need some council on this one since we can
            tune how much fragment this script can generate for one SMILES. Also, everything (line 69 to 77)
            needs to be under a for loop or else it will break (as in not generating the correct amount of fragments,
            usually much less than the actual amount). I'm not sure why
    :param smiles:
    :return:
    """
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(0, 4, fName)  # I need more research and tuning on this one
    fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
    fcgen = FragmentCatalog.FragCatGenerator()
    mol = Chem.MolFromSmiles(smiles)
    fcount = fcgen.AddFragsFromMol(mol, fcat)
    # print("This SMILES, %s, has %d fragments" % (smiles, fcount))
    frag_list = []
    for frag in range(fcount):
        frag_list.append(fcat.GetEntryDescription(frag))  # List of molecular fragments
    return frag_list


def insert_fragments(df_for_fragments, graph):

    fragment_query = Query(graph=graph).__fragment_query__()

    df_for_fragments = df_for_fragments[['smiles', 'fragments']]
    smiles_frags_dicts = df_for_fragments.to_dict('records')

    tx = graph.begin(autocommit=True)
    tx.evaluate(fragment_query, parameters={"rows": smiles_frags_dicts})

