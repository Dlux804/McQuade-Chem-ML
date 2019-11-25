import pandas as pd
from rdkit import Chem

def load_smiles(self, file, drop=True):
    """ Find SMILES in CSV.  Return DataFrame and Series of SMILES.

    Keyword Arguments
    drop -- Drop all other columns besides smiles and target. Default = True
    """
    csv = pd.read_csv(file)
    for i in csv.head(0):
        try:
            pd.DataFrame(list(map(Chem.MolFromSmiles, csv[i])))
            smiles_col = csv[i]
        #                molob_col = pd.DataFrame(molob, columns = 'molobj')
        except:# TypeError:
            pass
    # rename the column with SMILES to 'smiles'
    csv = csv.rename(columns={smiles_col.name: "smiles"})
    if drop: # drop all extra columns
        csv = csv[['smiles', self.target]]

    return csv, smiles_col

def ingest_test(self, k):
    print('Successfully imported ingest function.')

    return k + 5
