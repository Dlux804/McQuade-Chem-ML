import numpy as np
import pandas as pd

"""
Script to clean up the CMC data so that it is clean enough to compare to other datasets.
"""

df = pd.read_csv('cmc.csv')

# print(df.head(5))
# print(df.columns)

df = df.rename(columns={"Unnamed: 6": "Unit", "Additives": "add_conc", "Additives.1": "additive", "Temp ©": "T", 'Canonical SMILES': "canon_smiles"})

# print(df.columns)

df = df.drop(['Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], axis=1)

# convert room temp string to int 23
mask = (df['T'] == 'RM')  # & (data['column1'] > 90)
df['T'][mask] = 23

# convert T and CMC to integers instead of strings
df[["T", "CMC"]] = df[["T", "CMC"]].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=['canon_smiles', 'CMC', 'Unit'])

print(df.columns)

# drop everything that has an additive in it.
# no_add = df[pd.isnull(df['additive']), pd.isnull(df['add_conc'])]
no_add = df[pd.isnull(df['additive'])]
no_add = df[pd.isnull(df['add_conc'])]

cross = set(no_add['canon_smiles'].to_list())
print("There are {} unique molecules after filtering. ".format(len(cross)))

no_add.to_csv('cmc_noadd.csv')

dfT = no_add[no_add['T'].between(19, 26)]


cross = set(dfT['canon_smiles'].to_list())
print("There are {} unique molecules after filtering. ".format(len(cross)))

# mask = (df['T'] == 'RM') #& (data['column1'] > 90)
# new = df['T'][mask] = 23

# other = df.loc[(df.T == 'RM'), 'T'] = 23
# for i, row in enumerate(no_add.itertuples(), 1):  # iterate through dataframe
#     c = row.Index
#     # print('fu', c)
#     # no_add.loc[no_add.T == 20, 'T'] = "Matt"
#     print(no_add.T == 20)
#
#     # if no_add.loc[c, 'T'] == 50:
#     #     print('found room temp')
#     #     no_add.loc[c, 'T'] = 23

# print(no_add.head(10))

