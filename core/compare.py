import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from time import time
import pandas as pd
import shutil
import glob
import misc
import ingest
import analysis
import cirpy
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import itertools


"""
Goal is to compare the output of many models in a systematic, automatic manner.
"""

def merger():

    #import csv files from folders
    path = r'C:/Users/luxon/OneDrive/Research/McQuade/Projects/NSF/OKN/phase1/Work/ml-hte-results-20200207'  # will vary by OS, computer
    allFiles = glob.glob(path + "/*/*.csv")
    with open('hte-models-Master-Results.csv', 'wb') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                print(fname + " has been imported.")

    master = pd.read_csv('hte-models-Master-Results.csv')#, dtype={'exp': object})
    # print(master)

    short = master.drop(['pvaM', 'regressor', 'feature_list'], axis=1)
    print(short)

def datasize(dataset):
    """Cacluate dataset size by counting rows in CSV. Accept filename (string) return int.

    Flaw is the assumption that each row is a valid data set
    entry.  It is safe for our basic cases.
    """
    with misc.cd('../dataFiles/'): # move to dataset directory
        with open(dataset) as f:
            size = sum(1 for line in f) - 1  # remove one for header
            print('The {} dataset has {} entries in it.'.format(dataset, size))
            return size


# string = '../dataFiles/ESOL.csv'
# datasize(string)

def moloverlap(datasets):
    """
    Identifies molecules conserved across datasets.
    Accepts dictionary of datasets to be compared.  Keys are dataset file, values are chemicalID column.
    Returns dataframe of conserved molecules.
    """
    with misc.cd('../dataFiles/'):  # move to dataset directory

        # use dictionary comprehension and iterate over argument dictionary
        # create resolve chemID and create dataframe with ingest.resolveID
        # d = {dataset: ingest.resolveID(dataset, col) for dataset, col in datasets.items()}
        # this is slow because of cirpy.  It would be faster if it skipped ones already in smiles...

        d = {}  # empty dict to add created dataframes to
        for dataset, col in datasets.items():
            # try to load molobj from smiles
            df = pd.read_csv(dataset)
            print('Trying to get molecules from SMILES for', dataset, "...")
            # add column to DF for molecule object
            PandasTools.AddMoleculeColumnToFrame(df, col, 'Molecule', includeFingerprints=True)
            if df['Molecule'].isnull().values.any():  # if any ID failed to be converted to molobjects
                print(dataset, 'failed to create objects. Attempting resolve.')
                df = ingest.resolveID(dataset, col) # try to find smiles for the ID column
                # retry converting to molobject after resolution
                PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'Molecule', includeFingerprints=True)

            """ Now we should have an RDKit Molecule object for each entry.  To makes sure there are no
            synonymous representations, i.e non-canonical SMILES, over write 'smiles' column
            with canonical smiles from RDKit.
            """
            # didnt want to take the time to find a more effecient way to do this other than loop
            for i, row in enumerate(df.itertuples(), 1):  # iterate through dataframe
                c = row.Index
                df.loc[c, 'canon_smiles'] = Chem.MolToSmiles(df.loc[c, 'Molecule'])  # write canonical smiles

            d.update({dataset: df})

        for combo in itertools.combinations(d.keys(), 2):
            print("\nCombo:", combo)
            # create set of canon_smiles of dataframes from each data set using comprehension
            ddf = [set(d[x]["canon_smiles"]) for x in combo] # for explanation of this uncomment for loop below
            df_list = [d[x] for x in combo]

            # for explanation purposes, loop prints components
            # for x in combo:
            #     print("x: " ,x)
            #     print("d[x]:", d[x])
            #     print("d[x]['canon_smiles']", d[x]["canon_smiles"])


            # df_list = [df.loc[~df.index.duplicated(keep='first')] for df in df_list]

            # drop duplicated smiles in each data frame?
            df_list = [df.drop_duplicates(subset='canon_smiles') for df in df_list]
            df_list = [df.set_index('canon_smiles') for df in df_list]


            # df3.loc[~df3.index.duplicated(keep='first')]
            # print("df_list: ",df_list)
            # df.drop_duplicates(inplace=True)
            # df_cross = pd.concat(df_list, join='outer', axis=1)
            df_cross = pd.concat(df_list, axis=1, join='inner')  # combine the dataframes of interest
            df_cross = df_cross.loc[:, ~df_cross.columns.duplicated()]
            # need to drop repeated column keys such as "Molecule" and 'canon_smiles', 'smiles'

            # print("df_cross: ", df_cross)
            # print("df_list is:", df_list)


            # cross = list(set.intersection(*map(set, ddf))) # do the comparison via intersection
            # cross = df_cross['canon_smiles'].to_list()
            cross = list(df_cross.index.values)
            print("Verifying unique entries via canonical smiles:", len(cross) == len(set(cross)))
            print('There are {} overlapping molecules in the {} datasets.'.format(len(cross), combo))
            # print(cross,"\n")
            #
            # xdf = pd.DataFrame({'canon_smiles': cross})
            # PandasTools.AddMoleculeColumnToFrame(xdf, 'canon_smiles', 'Molecule', includeFingerprints=True)
            # print("xdf:", xdf)
            # ximage = PandasTools.FrameToGridImage(
            #     df_cross, column='Molecule',
            #     molsPerRow=6, subImgSize=(400,200),
            #     legends=[str(i+1) for i in range(len(cross))]
            # )
            # ximage.save('cross.png')

            """ So I have the molecules that overlap, but I also want the properties that are measured for them.
            How do I get the intersections, but also keep their measurements?  Pandas merge, i believe"""

            return df_cross






# likely will be useful for visualizing conserved molecules.
# rdkit.Chem.PandasTools.FrameToGridImage(frame, column='ROMol', legendsCol=None, **kwargs)

data = {
    # "pyridine_cas.csv": "CAS",
    # "pyridine_smi_1.csv": "smiles",
    # "pyridine_smi_2.csv": "smiles",
    "logP14k.csv": "SMILES",
    "ESOL.csv": "smiles",
    # "Lipophilicity-ID.csv": "smiles",
    # # "jak2_pic50.csv": "SMILES",
    # "water-energy.csv" : "smiles"
    # "pyridine_smi_3.csv" : "smiles"
}

xdf = moloverlap(data)

analysis.plotter(xdf['Kow'], xdf['water-sol'], filename='LogP vs LogS', xlabel='LogP', ylabel='LogS')



