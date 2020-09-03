import math
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from rdkit.Chem import PandasTools
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from core.storage.dictionary import target_name_grid


def impgraph(self):
    """
    Objective: Make a feature importance graph. I'm limiting this to only rf and gdb since only they have feature
    importance (I might need to double check on that). I'm also limiting this to only rdkit2d since the rest are only 0s
    and 1s
    """

    # Get numerical feature importances

    importances2 = self.estimator.feature_importances_  # used later for graph


    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.feature_list, list(importances2))]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # prepare importance data for export and graphing
    indicies = (-importances2).argsort()
    varimp = pd.DataFrame([], columns=['variable', 'importance'])
    varimp['variable'] = [self.feature_list[i] for i in indicies]
    varimp['importance'] = importances2[indicies]
    # Importance Bar Graph
    plt.rcParams['figure.figsize'] = [15, 9]

    # Set the style
    plt.style.use('bmh')

    # intiate plot (mwahaha)
    fig, ax = plt.subplots()
    plt.bar(varimp.index, varimp['importance'], orientation='vertical')

    # Tick labels for x axis
    plt.xticks(varimp.index, varimp['variable'], rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title(self.run_name + ' Variable Importances')

    # ax = plt.axes()
    ax.xaxis.grid(False)  # remove just xaxis grid

    plt.savefig(self.run_name + '_importance-graph.png')
    plt.close()
    # self.impgraph = plt
    self.varimp = varimp


def pva_graph(self, use_scaled=False):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """
    # Reuse function for scaled data
    if use_scaled:
        pva = self.scaled_predictions
    else:
        pva = self.predictions

    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    norm = cm.colors.Normalize(vmax=pva['pred_std'].max(), vmin=pva['pred_std'].min())
    x = pva['actual']
    y = pva['pred_avg']
    plt.scatter(x, y, c=pva['pred_std'], cmap='plasma', norm=norm, alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Uncertainty")

    # set axis limits
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]

    # TODO: add histograms on axes
    # # definitions for the axes
    # left, width = 0.1, 0.65
    # bottom, height = 0.1, 0.65
    # spacing = 0.005
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]
    # ax_histx = plt.axes()
    # ax_histx.tick_params(direction='in', labelbottom=False)
    # ax_histy = plt.axes()
    # ax_histy.tick_params(direction='in', labelleft=False)
    # binwidth = 0.025
    # lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    # bins = np.arange(-lim, lim + binwidth, binwidth)
    # ax_histx.hist(x, bins=bins)
    # ax_histy.hist(y, bins=bins, orientation='horizontal')

    # ax_histx.set_xlim(ax_scatter.get_xlim())
    # ax_histy.set_ylim(ax_scatter.get_ylim())
    # ------------------------------------------------

    # ax = plt.axes()
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    if use_scaled:
        plt.title(self.run_name + f' Predicted vs. Actual (scaled)')
    else:
        plt.title(self.run_name + f' Predicted vs. Actual')

    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    plt.plot([], [], ' ', label='MSE = %.3f' % mse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    if use_scaled:
        plt.savefig(self.run_name+'_' + f'PVA_scaled.png')
    else:
        plt.savefig(self.run_name + '_' + f'PVA.png')
    plt.close()
    # plt.show()
    # self.pva_graph = plt
    # return plt


def hist(self):

    def __plot__(name, data, plot_name):
        plt.style.use('bmh')
        # plt.grid(b=None)  # Get rid of grid lines
        plt.rcParams['axes.axisbelow'] = True  # Put grid lines behind bars of data
        plt.hist(data, bins=num_of_bins, histtype='step', rwidth=0.8, fill=True)
        plt.plot([], [], ' ', label=f'{plot_name}')
        plt.title(f'Histogram of {plot_name}')
        plt.xlabel(x_axis)
        plt.ylabel('Frequency')
        plt.savefig(self.run_name + '_' + f'hist_{name}.png')
        plt.close()

    # Use Doane's formula for number of bins https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    # Want to make sure all hist data uses the same number of bins
    num_of_bins = 0
    doane_data = self.predictions[['actual', 'pred_avg']]
    for column in doane_data.columns:
        data = doane_data[column]
        n = len(data)
        std = data.std()
        g = math.sqrt((6 * (n - 1)) / ((n + 1) * (n + 3)))
        np_bins = round(1 + math.log2(n) + math.log2(1 + abs(g) / std))
        if np_bins > num_of_bins:
            num_of_bins = np_bins

    # Increase bin for finer results
    num_of_bins = num_of_bins * 4

    x_axis = target_name_grid(self.dataset)
    plot_dict = {'predicted': self.predictions['pred_avg'],
                 'actual': self.predictions['actual'],
                 'predicted_scaled': self.scaled_predictions['pred_avg'],
                 'actual_scaled': self.scaled_predictions['actual']}
    for name, data in plot_dict.items():
        plot_name = " ".join(name.split('_'))
        __plot__(name, data, plot_name)


def plotter(x, y, filename=None, xlabel='', ylabel=''):
    """
    General plotting function for creating an XY scatter plot.
    Accepts x-axis data and y-axis data as (numpy or pd.Series or lists?)
    Returns graph object.  If filename keyword is given, will save to file (PNG)
    ____________________________
    Keyword Arguments
    filename:  None or string. Default = None.  Specify filename for saving to PNG file.  Do not include extension.
    xlabel: string. Default = ''.  X-axis label.
    ylabel: string. Default = ''.  Y-axis label.

    """

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    plt.plot(x, y, 'o')
    # ax = plt.axes()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(filename)
    # ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)
    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    if filename is not None:
        plt.savefig(filename + '.png')

    plt.show()


def grid_image(df, filename, molobj=True, smi='smiles'):  # list of molecules to print and substructre to align
    """
    Creates and saves grid image of 2D drawings of molecules.
    Accepts dataframe containing a column titled "Molecule" that contains RDKit molecule objects.
    Accepts filename as string (without .png) for image file.
    Returns nothing, saves file in current directory.
    _____________________________
    Keyword Arguments:
    molobj=True, if RDKit MolObj column exists in df.  (Must be headed "Molecule")
    smi='smiles', if molojb=False then use column titled smi to create MolObj column.

     """

    if not molobj:  # no molobj exists
        PandasTools.AddMoleculeColumnToFrame(df, smi, 'Molecule', includeFingerprints=True)

    # this code makes multiple images of n molecules.  May be prefered for large sets of molecules.

    # mols = df['Molecule']
    # # for every molecule
    # for mol in mols:
    #     # generate 2D structure
    #     Chem.Compute2DCoords(mol)
    #
    # n = 250  # number of structures per image file
    #
    # total = len(mols)  # number of molecules being printed
    #
    # # break list into printable sections of size n
    # mols = [mols[i: i + n] for i in range(0, total, n)]
    # subcount = 1  # counter for how many sections needed
    #
    # for i in mols:  # for every sublist of n molecules do...
    #     # make the images on grid
    #     img = Draw.MolsToGridImage(i, molsPerRow=6, subImgSize=(1500, 900), legends=[str(x) for x in range(total)])
    #
    #     # Save a sub image
    #     img.save('mole-grid-' + str(subcount) + '.png')
    #     subcount = subcount + 1

    # create images of molecules in dataframe
    mol_image = PandasTools.FrameToGridImage(
        df, column='Molecule',
        molsPerRow=3, subImgSize=(800, 400),
        legends=[str(i + 1) for i in range(len(df['Molecule']))]
    )
    mol_image.save(filename + '.png')  # shold use a better naming scheme to avoid overwrites.


def classification_graphs(self):
    """
    This function creates two graphs for single-label classification.

    The first block creates a roc_curve graph,
    while the second plot creates a precision/recall vs threshold graph.
     """

    if self.task_type == 'single_label_classification':
        # Creates and saves a graphical evaluation for single-label classification
        fpr, tpr, thresholds = roc_curve(self.test_target, self.predictions)
        plot_roc_curve(fpr, tpr, self.auc_avg, self.acc_avg, self.f1_score_avg)
        filename = "roc_curve.png"
        plt.legend(loc='best')
        plt.savefig(self.run_name + '_' + filename)
        plt.close()


        precisions, recalls, thresholds = precision_recall_curve(self.test_target, self.predictions)
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        filename3 = "precision_recall_vs_threshold.png"
        plt.legend(loc='best')
        plt.savefig(self.run_name + '_' + filename3)
        plt.close()


def plot_roc_curve(fpr, tpr, auc, acc, f1, label=None):
    """
This function is called by the classification_graphs function, and is used to create the roc_curve graph,
as well as to add labels showing import metrics to the graph.
     """
    plt.plot(fpr, tpr, linewidth=2, label="Roc_Auc_Score Average{{}} = {}".format(auc))
    plt.plot([0, 1], [0, 1], 'k--', label='y=x')

    plt.plot([], [], ' ', label='Accuracy_Score_Average = %.3f' % acc)
    plt.plot([], [], ' ', label='F1_Score_Average = %.3f' % f1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
This function is called by the classification_graphs function, and is used to create
a precision/recall vs threshold graph.
     """
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Threshold')
