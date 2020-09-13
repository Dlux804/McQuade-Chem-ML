import cirpy
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

        except Exception:
            pass
    # rename the column with SMILES to 'smiles'
    csv = csv.rename(columns={smiles_col.name: "smiles"})
    if drop:  # drop all extra columns
        csv = csv[['smiles', self.target_name]]

    return csv, smiles_col


def resolveID(file, column):  # TODO Consider incorporation of this function in load_csv()
    """ Resolves chemical ID using cripy package from NCI.
    Accepts csv file path and name (as string) and string of column header to be resolved.
    Returns dataframe with added column containing smiles."""

    df = pd.read_csv(file)  # read csv file

    # for i in df.head(0):  # look at all columns
    #     try:
    #         pd.DataFrame(list(map(Chem.MolFromSmiles, csv[i])))
    #         # pd.DataFrame(list(map(cirpy.resolve(,'smiles'), csv[i])))
    #         df[i].apply(cirpy.resolve, args=())
    #         s.apply(subtract_custom_value, args=(5,))
    #         from functools import partial
    #
    #         mapfunc = partial(my_function, ip=ip)
    #         map(mapfunc, volume_ids)
    #         smiles_col = csv[i]
    #
    #     except Exception:
    #         pass
    for i, row in enumerate(df.itertuples(), 1):  # iterate through dataframe
        c = row.Index
        id = df.loc[c, column]  # get cas number from df
        print('Resolving', id)

        # look up the CAS, convert to smiles
        df.loc[c, 'smiles'] = cirpy.resolve(id, 'smiles')  # store in df

        # provides output text
        if df.loc[c, 'smiles'] == None:
            print('No SMILES found')
            print()

        else:
            print('smiles found :)')
            print()

    # drop if smiles was not found
    df3 = df.dropna()
    print(df3.head(5))
    return df3
