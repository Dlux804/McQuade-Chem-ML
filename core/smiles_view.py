"""
Simple RDKit functions to re-write smiles and view molecules.
Mostly used for producing smiles of molecules for searching the GDB.
Alternatively, useful for visualizing compounds found in GDB.
-- Adam Luxon
"""

from rdkit import Chem
from rdkit.Chem import Draw

def canon_smi(smi):
    """Prints out SMILES.  Useful for going from PubChem version of SMILES to arromatic SMILES."""
    mol = Chem.MolFromSmiles(smi)
    smi  = Chem.MolToSmiles(mol)
    print(smi)
    return smi


def mol_view(smi):
    """Will pop-up image of molecule in image viewing software like MS Paint. """
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol)
    img.show()

smi = 'O=C(Cl)Oc1cccc(CCl)c1'
smi = canon_smi(smi)
mol_view(smi)