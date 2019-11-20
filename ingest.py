def findsmiles(file):
    ''' Find SMILES in CSV.  Return DataFrame and string of SMILES column label.'''
    csv = pd.read_csv(file)
    for i in csv.head(0):
        try:
            pd.DataFrame(list(map(Chem.MolFromSmiles, csv[i])))
            smiles_col = csv[i]
        #                molob_col = pd.DataFrame(molob, columns = 'molobj')
        except TypeError:
            pass
    return csv, smiles_col

def ingest_test():
    print('Successfully imported ingest function.')
