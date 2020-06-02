from Draw import drawer
from Neo4j.US_patents.backends import map_rxn_functional_groups

rxn_smarts = ''

drawer.save_reaction_image(rxn_smiles=rxn_smarts, location='example_reaction.png')
print(map_rxn_functional_groups(rxn_smarts, difference_only=True))
