from padelpy import from_smiles
import pandas as pd
# calculate molecular descriptors for propane
descriptors = from_smiles('CCC')
df = pd.DataFrame(descriptors, columns=descriptors.keys(), index=[0])
print(df)
# in addition to descriptors, calculate PubChem fingerprints
# desc_fp = from_smiles('CCC', fingerprints=True)

# only calculate fingerprints
# fingerprints = from_smiles('CCC', fingerprints=True, descriptors=False, output_csv='fingerprints.csv')

# save descriptors to a CSV file
# _ = from_smiles('CCC', output_csv='descriptors.csv')
