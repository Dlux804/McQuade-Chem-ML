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
    Objective: Get SMILES directly from pipeline
    Intent: I created this function only when the user is creating Neo4j graphs directly from the ML pipeline
    :param self:
    :return:
    """
    smiles_list = list(self.data['smiles'])
    return smiles_list


def canonical_smiles(smiles_list):
    """
    Objective: Create list of canonical SMILES from SMILES
    Intent: While the SMILES in dataset from moleculenet.ai are all canonical, it is always good to be safe. I don't
            know if I should add in a way to detect irregular SMILES and remove the rows that contains them in the
            dataframe. However, that process should be carried out at the start of the pipeline instead of at the end.
    :param smiles_list:
    :return:
    """
    canonical_smiles = list(map(Chem.MolToSmiles, list(map(Chem.MolFromSmiles, smiles_list))))  # SMILES to Canonical
    return canonical_smiles


def cypher_smiles_fragments(canonical_smiles, fragment_list):
    """
    Objective: Import SMILES and fragments into Neo4j with relationships based on our ontology
    Intent: I want a separate function just for importing SMILES nad fragments to Neo4j. While I can put this into the
            function "fragments_to_neo" located below, I think the code is more readable this way.
    :param canonical_smiles: A SMILES (*cough *cough CANONICAL SMILES *cough *cough)
    :param fragment_list: List of fragments for one SMILES
    :return:
    """
    for fragment in tqdm(fragment_list, desc="Importing fragments to Neo4j"):
        g.evaluate("merge (frag:Fragment {fragment: $fragment})", parameters={'fragment': fragment})
        g.evaluate("match (smiles:SMILES {SMILES:$mol}), (frag:Fragment {fragment: $fragment})"
                   "merge (frag)<-[:HAS_FRAGMENT]-(smiles)",
                   parameters={'fragment': fragment, 'mol': canonical_smiles})


def fragments_to_neo(canonical_smiles):
    """
    Objective: Create fragments and import them into Neo4j based on our ontology
    Intent: This script is based on Adam's "mol_frag.ipynb" file in his deepml branch, which is based on rdkit's
            https://www.rdkit.org/docs/GettingStartedInPython.html. I still need some council on this one since we can
            tune how much fragment this script can generate for one SMILES. Also, everything (line 69 to 77)
            needs to be under a for loop or else it will break (as in not generating the correct amount of fragments,
            usually much less than the actual amount). I'm not sure why
    :param canonical_smiles:
    :return:
    """

    for smiles in tqdm(canonical_smiles, desc="Creating molecular fragments for SMILES"):
        fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
        fparams = FragmentCatalog.FragCatParams(0, 3, fName)  # I need more research and tuning on this one
        fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
        fcgen = FragmentCatalog.FragCatGenerator()
        mol = Chem.MolFromSmiles(smiles)
        fcount = fcgen.AddFragsFromMol(mol, fcat)
        print(f"This SMILES, {smiles}, has {fcount} fragments")
        fcgen.AddFragsFromMol(mol, fcat)
        frag_list = []
        for frag in range(fcount):
            frag_list.append(fcat.GetEntryDescription(frag))  # List of molecular fragments
        cypher_smiles_fragments(smiles, frag_list)
