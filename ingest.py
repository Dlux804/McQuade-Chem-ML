import pandas as pd
from rdkit import Chem

def load_smiles(self, file):
    ''' Find SMILES in CSV.  Return DataFrame and Series of SMILES.'''
    csv = pd.read_csv(file)
    for i in csv.head(0):
        try:
            pd.DataFrame(list(map(Chem.MolFromSmiles, csv[i])))
            smiles_col = csv[i]
        #                molob_col = pd.DataFrame(molob, columns = 'molobj')
        except:# TypeError:
            pass

    return csv, smiles_col

def ingest_test(self, k):
    print('Successfully imported ingest function.')

    return k + 5
