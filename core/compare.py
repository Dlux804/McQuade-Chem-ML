import shutil
import glob
import itertools
import matplotlib.pyplot as plt
from functools import reduce

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from core.ingest import resolveID
from core.storage import cd



"""
Goal is to compare the output of many models in a systematic, automatic manner.
"""


def datasize(dataset):
    """Cacluate dataset size by counting rows in CSV. Accept filename (string) return int.

    Flaw is the assumption that each row is a valid data set
    entry.  It is safe for our basic cases.
    """
    with cd('../dataFiles/'): # move to dataset directory
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
    with open('hte-models-Master-results.csv', 'wb+') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                # print(fname + " has been imported.")

    mdf = pd.read_csv('hte-models-Master-results.csv')#, dtype={'exp': object})
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
        #

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

            X = np.arange(len(row[0]))
            pos = [x - (1 - space) / 2. + i * width for x in X]

            # from http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/

            rects = plt.bar(pos, row[0], yerr=row[1],
                            width=width, capsize=10, label=row[2][1])  # ,

            autolabel(rects)


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





