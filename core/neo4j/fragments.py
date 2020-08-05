"""
Objectvie: Create molecular fragments, import them into Neo4j and add them into our ontology
"""

import os
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit import Chem
from math import ceil
from tqdm import tqdm
from py2neo import Graph
import pandas as pd
# import py2neo
# Connect to Neo4j Destop.

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


def fragments_to_neo(smiles):
    """
    Objective: Create fragments and import them into Neo4j based on our ontology
    Intent: This script is based on Adam's "mol_frag.ipynb" file in his deepml branch, which is based on rdkit's
            https://www.rdkit.org/docs/GettingStartedInPython.html. I still need some council on this one since we can
            tune how much fragment this script can generate for one SMILES. Also, everything (line 69 to 77)
            needs to be under a for loop or else it will break (as in not generating the correct amount of fragments,
            usually much less than the actual amount). I'm not sure why
    :param g:
    :param smiles:
    :return:
    """

    # mol_feat_query = """
    #     UNWIND $fragments as fragment
    #     MERGE (mol:Molecule {SMILES: fragment.smiles})
    #         FOREACH (value in fragment.fragments|
    #             MERGE (fragment:Fragments {name: value.fragments})
    #             MERGE (mol)-[:HAS_FRAGMENTS]->(fragment)
    #                 )
    #     """

    # mol_feat_query = """
    #         UNWIND $fragments as fragment
    #         With fragment
    #         CALL apoc.periodic.iterate('
    #         MERGE (mol:Molecule {SMILES: $frags.smiles})
    #             FOREACH (value in $frags.fragments|
    #                 MERGE (fragment:Fragments {name: value.fragments})
    #                 MERGE (mol)-[:HAS_FRAGMENTS]->(fragment)
    #                     )
    #                     ',';',
    #         {batchSize:100000, parallel:true, params:{frags:fragment}}) YIELD batches, total
    #     RETURN batches, total
    #         """
    # smiles = str(row['smiles'])

    # smiles = str(row['smiles'])

    # for smiles in tqdm(canonical_smiles, desc="Creating molecular fragments for SMILES"):
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


def insert_fragments(temp_df, graph):

    mol_feat_query = """
            UNWIND $rows as row
                MERGE (mol:Molecule {SMILES: $row.smiles})
                FOREACH (fragment in $row.fragments |
                    MERGE (frag:Fragments {name: fragment})
                    MERGE (mol)-[:HAS_FRAGMENTS]->(frag)
                    )
            """

    temp_df = temp_df[['smiles', 'fragments']]
    smiles_frags_dicts = temp_df.to_dict('records')

    batch = 500

    range_molecules = []
    for index, smiles_frag_dict in tqdm(enumerate(smiles_frags_dicts), total=len(smiles_frags_dicts)):
        range_molecules.append(smiles_frag_dict)
        if index % batch == 0:
            tx = graph.begin(autocommit=True)
            tx.evaluate(mol_feat_query, parameters={"rows": range_molecules})
            range_molecules = []
    if range_molecules:
        tx = graph.begin(autocommit=True)
        tx.evaluate(mol_feat_query, parameters={"rows": range_molecules})
