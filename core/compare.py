import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
import pprint


"""
Goal is to compare the output of many models in a systematic, automatic manner.
"""


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


def merger():
    """ Function to combine all results csv  into a single file. """

    #import csv files from folders
    path = r'C:/Users/luxon/OneDrive/Research/McQuade/Projects/NSF/OKN/phase1/Work/ml-hte-results-20200207'  # Adam's tablet. will vary by OS, computer
    # path = r'C:/Users/Adam/OneDrive/Research/McQuade/Projects/NSF/OKN/phase1/Work/ml-hte-results-20200207' # Adam's desktop
    allFiles = glob.glob(path + "/*/*.csv")
    with open('hte-models-Master-Results.csv', 'wb+') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                # print(fname + " has been imported.")

    mdf = pd.read_csv('hte-models-Master-Results.csv')#, dtype={'exp': object})
    # print(master)

    # grab unique datasets into a set
    dsets = set(mdf['dataset'])

    # create dictionary of datasets and their size
    sizes = {}
    for data in dsets:
        sizes.update({data: datasize(data)})

    # map the dictionary to create a new column of dataset size
    mdf['data_size'] = mdf['dataset'].map(sizes)
    # print(mdf.head(3))

    # short = mdf.drop(['pvaM', 'regressor', 'feature_list'], axis=1)
    # print(short)

    return mdf

# mdf = merger()



# string = 'ESOL.csv'
# datasize(string)

# obsolete
def model_compare(master_csv):
    """
    Specific function for analyzing and comparing the outputs of ML models.
    Accepts a csv of the compiled model results [output of merger()]
    """
    # import
    mdf = pd.read_csv(master_csv)

    # grab unique datasets into a set
    dsets = set(mdf['dataset'])

    # create dictionary of datasets and their size
    sizes = {}
    for data in dsets:
        sizes.update({data : datasize(data)})

    # map the dictionary to create a new column of dataset size
    mdf['data_size'] = mdf['dataset'].map(sizes)
    print(mdf.head(3))


def alg_vs_acc(df):
    """
    Graphing function for comparing model algorithm vs accuracy.
    Produceds one graph per each data set in passed dataframe.
    Graph will have algorithm on x-axis, RMSE on y-axis.  Bar color represents featurization methods.
    """
    df = df.filter(['algorithm', 'dataset', 'feat_meth', 'rmse_avg', 'rmse_std'])

    # just for trouble shooting and dev, filter
    df = df[df['dataset'] != 'logP14k.csv']

    datasets = list(set(df['dataset']))
    print("\nThe imported dataframe contains entries for the following {} datasets: {}".format(len(datasets), datasets))

    # print(df)

    for data in datasets:  # loop through each dataset
        print('\nUsing:', data)
        # filter df ML runs that used particular dataset
        df_new = df[df['dataset'] == data]
        # print(df_new)

        # grab featurization methods used
        feats = list(set(list(df_new['feat_meth'])))
        # len(feats) is how many bars there will be per algorithm
        algs = list(set(df_new['algorithm']))
        # print("Algorithms set: ", algs)
        # len(algs) is how many groups (xticks) of bars there will be.
        # data for graph will be an len(feats) by len(algs) array.
        # data[0] will be a 1D array containing the rmse values for a featurization method across alg types

        errmat = []  # empty list that we will append rmse arrays to for each feat method
        stdmat = []  # matrix for std of rmse
        di = {}

        for feat in feats: # for every featurization method used on this dataset...
            # print("Featurizations:", feats)
            # get the RMSE value and put in a list
            # use .query to filter df for feat, then grab rmse column
            rmse = list(df_new.query('feat_meth == @feat')['rmse_avg'])  # use @ to ref a variable
            std = list(df_new.query('feat_meth == @feat')['rmse_std'])
            alg =  list(df_new.query('feat_meth == @feat')['algorithm'])
            meth = list(df_new.query('feat_meth == @feat')['feat_meth'])
            # print("rmse values: ", rmse)
            d = dict(zip(algs, rmse))
            """
            Using a dictionary because sets lose order and rmse values have no promise to be 
            ordered in the same way as the algorithms.  Then I nested the dictonaries because otherwise, the 
            key:value pairs would be overwritten for each iteration through the featurization methods 
            (algorithm is key and RMSE is value).
            
            This seems like its going to be a nightmare to to put into matplotlib.  Also, this level of structure
            pretty much brings us back to a dataframe... Is there an easier way to access the specific data I want
            for each series or bars directly from the dataframe instead of pulling it out of the data frame and 
            restructuring it?  
            """
            # print('RMSE for data set {} using featurization {} is: {}'.format(data, feat, rmse))

            # make a list of lists containing rmse and std
            info = [rmse, std, meth]
            # add RMSE data to the rmse errmat
            errmat.append(info)


            di.update({feat:d})
            # print("RMSE errmat:", errmat)
            # print("RMSE Dictonary after adding data for {}: ".format(feat), di)
            print()



        # start graphing
        # https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781849513265/1/ch01lvl1sec16/plotting-multiple-bar-charts
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        labels = alg
        gap = 1 / len(errmat)
        space = 0.5 / len(errmat)
        # gap = 0.3
        width = (1 - space) / (len(errmat))  # from http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
        # indeces = range(1, len(alg) + 1)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""

            # Get y-axis height to calculate label position from.
            (y_bottom, y_top) = ax.get_ylim()
            y_height = y_top - y_bottom
            i = 0
            for rect in rects:
                # print("Rect: ", rect)
                height = rect.get_height()
                err = rects.errorbar.lines[1][1].get_data()[1][i]
                i += 1
                # print("upper bound of error: ", err)
                label_position = err + (y_height * 0.01)
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, label_position),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            size=12)

        for i, row in enumerate(errmat):
            # print(row)
            # print("rmse list", row[0])
            # print(alg, i)
            X = np.arange(len(row[0]))
            pos = [x - (1 - space) / 2. + i * width for x in X]
            # print(pos)
            # from http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/

            # rects = plt.bar(X + i * gap, row,
            #         width=gap, label=feats[i-1], align='center')  #,
            #         # color=color_list[i % len(color_list)])
            rects = plt.bar(pos, row[0], yerr=row[1],
                            width=width, capsize=10, label=row[2][1])  # ,
            # print("Rects created: ", rects)
            # for rect in rects:
                # err = rect.errorbar.lines[1][1].get_data()[1]
                # print("Did I get the error?", err)
            # print("Error bar lines ydata for upper bound:", rects.errorbar.lines[1][1].get_data()[1])
            # color=color_list[i % len(color_list)])
            autolabel(rects)
            # rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
            # rects2 = ax.bar(x + width / 2, women_means, width, label='Women')

        x = np.arange(len(labels))  # the label locations
        ax.set_title('{}: Algorithm vs RMSE'.format(data), fontsize=20)
        ax.set_xticks(x)
        ax.tick_params(axis='both', which='major', labelsize=18)
        # ax.set_xticks(indeces)
        ax.set_xticklabels(labels)
        plt.xlabel('Learning Algorithm', fontsize=18)
        plt.ylabel('RMSE', fontsize=18)
        ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)



        plt.show()
        plt.close()

        # each feat method will need to have an array of RMSE values
        # order must match the order of algorithms (x ticks)
        """
        # plotting bars from a dict
        plt.bar(range(len(D)), list(D.values()), align='center')
        plt.xticks(range(len(D)), list(D.keys()))
        """


    """
    # bar plot orange and blue
    labels = set(df['algorithm'])
    print(len(labels))
    men_means = [20, 34, 30, 35]
    women_means = [25, 32, 34, 20]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
    rects2 = ax.bar(x + width / 2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        "Attach a text label above each bar in *rects*, displaying its height."
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
    """

# alg_vs_acc(mdf)


'Code for stacked 3D bar plot'
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
#
# ax.set_xlabel("Algorithm")
# ax.set_ylabel("Featurization")
# ax.set_zlabel("RMSE")
# # ax.set_xlim3d(0,10)
# # ax.set_ylim3d(0,10)
#
# # xpos = [2,5,8,2,5,8,2,5,8]
# xpos = df['algorithm']
# # ypos = [1,1,1,5,5,5,9,9,9]
# ypos = df['feat_meth']
# # zpos = np.zeros(9)
# zpos = np.zeros(df.shape[0])
#
# dx = np.ones(9)
# dy = np.ones(9)
# dz = [np.random.random(9) for i in range(4)]  # the heights of the 4 bar sets
#
# _zpos = zpos   # the starting zpos for each bar
# colors = ['r', 'b', 'g', 'y']
# for i in range(4):
#     ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color=colors[i])
#     _zpos += dz[i]    # add the height of each bar to know where to start the next
#
# plt.gca().invert_xaxis()
# plt.show()

def moloverlap(datasets, n, image=False):
    """
    Identifies molecules conserved across datasets.
    Accepts dictionary of datasets to be compared.  Keys are dataset file, values are chemicalID column.
    Accepts number of datasets to overlap at a time.
    Returns dataframe of conserved molecules.

    Keyword Arguments:
    ______________________
    image=False,  Whether to create a grid image of the overlapping molecules.

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

    for combo in itertools.combinations(d.keys(), n):
        print("\nCombo:", combo)
        # create set of canon_smiles of dataframes from each data set using comprehension
        # ddf = [set(d[x]["canon_smiles"]) for x in combo] # for explanation of this uncomment for loop below
        df_list = [d[x] for x in combo]

        # for explanation purposes, loop prints components
        # for x in combo:
        #     print("x: " ,x)
        #     print("d[x]:", d[x])
        #     print("d[x]['canon_smiles']", d[x]["canon_smiles"])


        # drop duplicated smiles in each data frame just in case
        df_list = [df.drop_duplicates(subset='canon_smiles') for df in df_list]

        # Set index of df to 'canon_smiles' before concat
        df_list = [df.set_index('canon_smiles') for df in df_list]

        # concat with 'inner' will keep only overlapping index.
        df_cross = pd.concat(df_list, axis=1, join='inner')  # combine the dataframes of interest

        # need to drop repeated column keys such as "Molecule" and 'canon_smiles', 'smiles'
        df_cross = df_cross.loc[:, ~df_cross.columns.duplicated()]


        # print("df_cross: ", df_cross)
        # print("df_list is:", df_list)
        # cross = list(set.intersection(*map(set, ddf))) # do the comparison via intersection
        # cross = df_cross['canon_smiles'].to_list()

        cross = list(df_cross.index.values)
        print("Verifying unique entries via canonical smiles:", len(cross) == len(set(cross)))
        print('There are {} overlapping molecules in the {} datasets.'.format(len(cross), combo))
        print(cross,"\n")
        if image:
            # create images of molecules that overlap
            ximage = PandasTools.FrameToGridImage(
                df_cross, column='Molecule',
                molsPerRow=6, subImgSize=(400,200),
                legends=[str(i+1) for i in range(len(cross))]
            )
            ximage.save('cross.png') # shold use a better naming scheme to avoid overwrites.


    return df_cross




data = {
    # "pyridine_cas.csv": "CAS",
    # "pyridine_smi_1.csv": "smiles",
    # "pyridine_smi_2.csv": "smiles",
    # "cmc_noadd.csv": "canon_smiles",
    # "logP14k.csv": "SMILES",
    "18k-logP.csv": "smiles",
    "ESOL.csv": "smiles",
    "cmc_smiles_26.csv": "smiles",
    "flashpoint.csv": "smiles",
    # "Lipophilicity-ID.csv": "smiles",
    # # "jak2_pic50.csv": "SMILES",
    # "water-energy.csv" : "smiles"
    # "pyridine_smi_3.csv" : "smiles"
}

# xdf = moloverlap(data,2)
# analysis.plotter(xdf['Kow'], xdf['water-sol'], filename='LogP vs LogS', xlabel='LogP', ylabel='LogS')




with misc.cd('../dataFiles/'): # move to dataset directory
    cmc = pd.read_csv('cmc_smiles_26.csv')
    cmc = cmc[:9]
    print(cmc)
    analysis.grid_image(cmc, 'cmc_molecules_short', molobj=False)
