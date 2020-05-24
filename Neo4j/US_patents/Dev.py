from rdkit.Chem import MolToSmiles, MolFromSmiles, MolToSmarts, MolFromSmarts, RDConfig, FragmentCatalog, rdChemReactions
from rdkit.Chem import Draw

from Neo4j.US_patents import US_patents_xml_to_csv
from Neo4j.US_patents.backends import save_reaction_image, get_fragments


smiles = "c1ccccc1O"

frags = get_fragments(smiles)
print(frags)

