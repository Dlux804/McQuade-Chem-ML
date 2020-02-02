from padelpy import from_smiles
import pandas as pd
import os

"""
Objective: We want to test if the example codes given in padel's github actually works. We also want to understand the
ins and outs of this module. 
Everytime the code is run, a .smi file is formed
"""

current_dir = "C:/Users/quang/McQuade-Chem-ML/padel/example"
# calculate molecular descriptors for propane
descriptors = from_smiles('C1=CC=CC=C1', output_csv='descriptors.csv')
df = pd.DataFrame(descriptors, columns=descriptors.keys(), index=[0])
print(df)
# for file in os.listdir(current_dir):
#     if file.endswith(".smi"):
#         os.remove(os.path.join(current_dir, file))
# in addition to descriptors, calculate PubChem fingerprints
# desc_fp = from_smiles('CCC', fingerprints=True)

# only calculate fingerprints
fingerprints = from_smiles('CCC', fingerprints=True, descriptors=False, output_csv='fingerprints.csv')

# save descriptors to a CSV file
# _ = from_smiles('CCC', output_csv='descriptors.csv')
