from rdkit.Chem import MolFromSmiles, MolFromSmarts
import pathlib
import pandas as pd

from Neo4j.US_patents.backends import map_rxn_functional_groups, save_reaction_image

rxn_smiles = ("[C&H2:1](-,:[N&H2:13])[C&H2:2][C&H2:3][C&H2:4][C&H2:5][C&H2:6][C&H2:7][C&H2:8][C&H2:9][C&H2:10][C&H2:11][C&H3:12].[C:14](-,:[O&H1:18])(=[O:17])[C&H1:15]=[C&H2:16].[C&H2:19]=O>>[C&H2:1](-,:[N:13]1[C&H1:16]=[C&H1:15][C:14](=[O:18])[O:17][C&H2:19]1)[C&H2:2][C&H2:3][C&H2:4][C&H2:5][C&H2:6][C&H2:7][C&H2:8][C&H2:9][C&H2:10][C&H2:11][C&H3:12]")

print(map_rxn_functional_groups(rxn_smiles, difference_only=True))
save_reaction_image(rxn_smiles, directory_location='reactions', svg=True)
