import pandas as pd
from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import PandasTools

"""
Objective: Find Similar SMILES fragments and use the similarities to make knowledge graphs.

You know, big boys big girls stuff
"""
# Read csv
all_df = pd.read_csv('short_freesolv.csv')
# print(all_df)
# Make molob using PandasTools
df = PandasTools.AddMoleculeColumnToFrame(all_df, 'smiles', 'molobj', includeFingerprints=True)
print(all_df)
molob = all_df['molobj']
