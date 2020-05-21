from rdkit.Chem import MolToSmiles, MolFromSmiles, MolToSmarts, MolFromSmarts, RDConfig, FragmentCatalog, rdChemReactions
from rdkit.Chem import Draw

from Neo4j.US_patents import US_patents_xml_to_csv
from Neo4j.US_patents.backends import save_reaction_image

head = 'C:/xampp/htdocs/reactions'

smiles = """
[C:1]([C:5]1[CH:10]=[CH:9][C:8]([OH:11])=[CH:7][CH:6]=1)([CH3:4])([CH3:3])[CH3:2]>[Ni]>[C:1]([CH:5]1[CH2:6][CH2:7][CH:8]([OH:11])[CH2:9][CH2:10]1)([CH3:4])([CH3:2])[CH3:3]
"""

save_reaction_image(smiles, head)


