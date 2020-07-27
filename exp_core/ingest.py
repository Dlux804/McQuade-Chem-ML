"""
Objective: Find SMILES column from a CSV, resolve Molecular IDs that aren't SMILES
"""

import pandas as pd
from rdkit import Chem
import cirpy


class Ingest:
    def __init__(self, dataset, target):
        """

        :param dataset: csv file (str)
        :param target: Header of target column
        :param ID_column: Column with IDs that aren't SMILES
        """
        self.dataset = dataset
        self.target = target
        self.data = pd.read_csv(self.dataset)

    def load_smiles(self, drop=True):
        """"""
        for i in self.data.head(0):
            try:
                pd.DataFrame(list(map(Chem.MolFromSmiles, self.data[i])))
                smiles_col = self.data[i]
            except Exception:
                pass
        # rename the column with SMILES to 'smiles'
        self.data = self.data.rename(columns={smiles_col.name: "smiles"})
        if drop:  # drop all extra columns
            self.data = self.data[['smiles', self.target]]

    def resolveID(self, ID_column):  # TODO Consider incorporation of this function in load_csv()
        """
        Resolves chemical ID using cripy package from NCI.
        Accepts csv file path and name (as string) and string of column header to be resolved.
        Returns dataframe with added column containing smiles.
        """

        df = pd.read_csv(self.dataset)  # read csv file

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
            id = df.loc[c, ID_column]  # get cas number from df
            print('Resolving', id)

            # look up the CAS, convert to smiles
            df.loc[c, 'smiles'] = cirpy.resolve(id, 'smiles')  # store in df

            # provides output text
            if df.loc[c, 'smiles'] is None:
                print('No SMILES found')
                print()

            else:
                print('smiles found :)')
                print()

        # drop if smiles was not found
        df3 = df.dropna()
        print(df3.head(5))
        return df3
