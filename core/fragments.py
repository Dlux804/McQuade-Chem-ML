"""
Objectvie: Create molecular fragments, import them into Neo4j and add them into our ontology
"""

import os
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit import Chem


def self_smiles(self):
    """
    Get SMILES directly from pipeline
    :param self:
    :return:
    """
    smiles_list = list(self.data['smiles'])
    return smiles_list


def canonical_smiles(smiles_list):
    canonical_smiles = list(map(Chem.MolToSmiles, list(map(Chem.MolFromSmiles, smiles_list))))  # SMILES to Canonical
    for smiles in canonical_smiles:
        yield smiles


def generate_fragments(smiles_list):
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(0, 1, fName)  # I need more research and tuning on this one
    fparams.GetNumFuncGroups()

    fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
    fcgen = FragmentCatalog.FragCatGenerator()
    canonical_smiles = list(map(Chem.MolToSmiles, list(map(Chem.MolFromSmiles, smiles_list))))  # SMILES to Canonical
    for smiles in canonical_smiles:
        fcount = fcgen.AddFragsFromMol(smiles, fcat)
        fcgen.AddFragsFromMol(smiles, fcat)

