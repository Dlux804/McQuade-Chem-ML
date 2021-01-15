"""
Simple RDKit functions to re-write smiles and view molecules.
Mostly used for producing smiles of molecules for searching our datasets.
Alternatively, useful for visualizing compounds.
-- Adam Luxon, VCU
"""

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

def canon_smi(smi):
    """Prints out SMILES.  Useful for going from PubChem version of SMILES to arromatic SMILES."""
    mol = Chem.MolFromSmiles(smi)
    smi  = Chem.MolToSmiles(mol)
    Chem.Kekulize(mol)
    smi_k = Chem.MolToSmiles(mol,kekuleSmiles=True)
    print('Non Kekulized SMILES: ', smi)
    print('Kekulized SMILES: ', smi_k)
    return smi


def mol_view(smi):
    """Will pop-up image (PNG) of molecule in image viewing software like MS Paint. """
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol)
    img.show()


def mol_svg(smi, filename=None, filepath='./', molSize=(450,150)):
    """Will write molecule 2D drawing to SVG file"""
    # if no filename given, use SMILES
    if not filename:
        name = smi + '.svg'
    else:
        name = filename + '.svg'

    mol = Chem.MolFromSmiles(smi)  # read the smiles string to RDKit
    d = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])  # set size of SVG image
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)  # draw molecule
    svg = d.GetDrawingText()  # get XML text of the SVG
    svg = svg + '</svg>'  # fix missing line to close svg

    # edit XML to create transparent background
    svg = svg.replace("rect style='opacity:1.0;fill:#FFFFFF", "rect style='opacity:1.0;fill:none", 1)

    # write to file
    with open(filepath + name, 'w') as f:
        f.write(svg)


# specify smiles here.  Could re-write to accept CLI input.
smi = "CCCCCCCCCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)NCC(=O)N[C@@H](CCCN)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H](C)C(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@H](CO)C(=O)N[C@@H]([C@H](C)CC(=O)O)C(=O)N[C@@H](CC(=O)c2ccccc2N)C(=O)O[C@@H]1C"
# rewrite smiles to canonical non-kelu
smi = canon_smi(smi)
mol_view(smi)
mol_svg(smi, "macro-cycle-monster")
