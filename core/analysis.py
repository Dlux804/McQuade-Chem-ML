import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from time import time
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from core import features


def predict(regressor, train_features, test_features, train_target, test_target):
    """Fit model and predict target values.  Return data frame of actual and predicted
    values as well as model fit time.
    regressor needs to have the '()' with it.  i.e it should be function(), not function"""
    start_time = time()

    regressor.fit(train_features, train_target)
    done_time = time()
    fit_time = done_time - start_time
    print('Finished Training After ',(done_time-start_time),"sec\n")
    # Make predictions
    predictions = regressor.predict(test_features)  # Val predictions

    true = test_target
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = true
    pva['predicted'] = predictions
#    pva.to_csv(exp+expt+'-pva_data.csv')

    return pva, fit_time


# Fearture importance class
def impgraph(model_name, regressor, train_features, train_target, feature_list):
    """
    Objective: Make a feature importance graph. I'm limiting this to only rf and gdb since only they have feature
    importance (I might need to double check on that). I'm also limiting this to only rdkit2d since the rest are only 0s
    and 1s
    """

    regressor.fit(train_features, train_target)
    # Get numerical feature importances
    importances2 = regressor.feature_importances_  # used later for graph

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, list(importances2))]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # prepare importance data for export and graphing
    indicies = (-importances2).argsort()
    varimp = pd.DataFrame([], columns=['variable', 'importance'])
    varimp['variable'] = [feature_list[i] for i in indicies]
    varimp['importance'] = importances2[indicies]
    # varimp.to_csv(exp + '-varimp.csv')
    # Importance Bar Graph
    plt.rcParams['figure.figsize'] = [15, 9]
    # print(varimp)
    # Set the style
    plt.style.use('bmh')

    #     # list of x locations for plotting
    #     x_values = list(range(importances.shape[0]))

    # intiate plot (mwahaha)
    fig, ax = plt.subplots()
    plt.bar(varimp.index, varimp['importance'], orientation='vertical')

    # Tick labels for x axis
    # plt.xticks(x_values, feature_list, rotation='vertical')
    plt.xticks(varimp.index, varimp['variable'], rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    # plt.title('EXP:' + exp + '  Variable Importances');

    # ax = plt.axes()
    ax.xaxis.grid(False)  # remove just xaxis grid

    plt.savefig(model_name + '-importance.png')
    return plt, varimp
    # else:
    #     pass


def replicate_multi(regressor, train_features, test_features, train_target, test_target, n=5):
    """
    Objective: Run model n times. Return dictionary of r2, mse and rmse average and standard deviation and predict each
    point multiple times to calculate uncertainty.

    :param regressor: Regression model
    :param n: Number of times the model is run
    :return: stats: Dictionary of valuable values, pva_multi: dataframe of predicted vs actual recorded 5 times,
            t: Array that contains running time.
    """
    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    t = np.empty(n)
    start_time = time()
    # create dataframe for multipredict
    pva_multi = pd.DataFrame([])
    for i in range(0, n):  # run model n times
        regressor.fit(train_features, train_target)

        # Make predictions
        predictions = regressor.predict(test_features)
        done_time = time()
        fit_time = done_time - start_time
        # Target data
        true = test_target
        # Dataframe for replicate_model
        pva = pd.DataFrame([], columns=['actual', 'predicted'])
        pva['actual'] = true
        pva['predicted'] = predictions
        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
        t[i] = fit_time
        # store as enumerated column for multipredict
        pva_multi['predicted' + str(i)] = predictions

    # done_time = time()
    # fit_time = done_time - start_time
    # t = fit_time
    pva_multi['pred_avg'] = pva.mean(axis=1)
    pva_multi['pred_std'] = pva.std(axis=1)
    pva_multi['actual'] = test_target
    stats = {
        'r2_avg': r2.mean(),
        'r2_std': r2.std(),
        'mse_avg': mse.mean(),
        'mse_std': mse.std(),
        'rmse_avg': rmse.mean(),
        'rmse_std': rmse.std(),
        'time_avg': t.mean(),
        'time_std': t.std()
    }
    print('Average R^2 = %.3f' % stats['r2_avg'], '+- %.3f' % stats['r2_std'])
    print('Average RMSE = %.3f' % stats['rmse_avg'], '+- %.3f' % stats['rmse_std'])
    print()

    return stats, pva_multi, t

def replicate_model(self, n):
    """Run model n times.  Return dictionary of r2, mse and rmse average and standard deviation."""

    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    t = np.empty(n)
    for i in range(0, n): # run model n times
        train_features, test_features, train_target, test_target, feature_list = features.targets_features(self.data, self.target, random=None)
        pva , fit_time = predict(self.regressor, train_features, test_features, train_target, test_target)

        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
        t[i] = fit_time

    stats = {
        'r2_avg': r2.mean(),
        'r2_std': r2.std(),
        'mse_avg': mse.mean(),
        'mse_std': mse.std(),
        'rmse_avg': rmse.mean(),
        'rmse_std': rmse.std(),
        'time_avg': t.mean(),
        'time_std': t.std()
    }

    print('Average R^2 = %.3f' % stats['r2_avg'], '+- %.3f' % stats['r2_std'])
    print('Average RMSE = %.3f' % stats['rmse_avg'], '+- %.3f' % stats['rmse_std'])
    print()
    return stats


def multipredict(regressor, train_features, test_features, train_target, test_target, n=5):
    """Predict each point multiple times to calculate uncertainty."""

    start_time = time()
    # create dataframe for predicted values
    pva = pd.DataFrame([])


    for i in range(0,n): # loop n times

        regressor.fit(train_features, train_target)

        # Make predictions
        predictions = regressor.predict(test_features)

        # store as enumerated column
        pva['predicted'+str(i)] = predictions

    # pva.to_csv(exp+expt+'-pva_data.csv')
    done_time = time()
    fit_time = done_time - start_time

    pva['pred_avg'] = pva.mean(axis=1)
    pva['pred_std'] = pva.std(axis=1)
    pva['actual'] = test_target

    return pva, fit_time


def pvaM_graphs(pvaM):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """
    r2 = r2_score(pvaM['actual'], pvaM['pred_avg'])
    mse = mean_squared_error(pvaM['actual'], pvaM['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pvaM['actual'], pvaM['pred_avg']))

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    norm = cm.colors.Normalize(vmax=pvaM['pred_std'].max(), vmin=pvaM['pred_std'].min())
    x = pvaM['actual']
    y = pvaM['pred_avg']
    plt.scatter(x, y, c=pvaM['pred_std'], cmap='plasma', norm=norm, alpha=0.7)
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
    plt.title('EXP: COLOR')  # TODO: Update naming scheme

    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    # plt.savefig(model_name+'-' +'.png')
    # plt.show()
    return plt


def pva_graphs(pva, model_name):
    """ Creates Predicted vs. Actual graph from predicted data. """
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
    print('R^2 = %.3f' % r2)
    print('MSE = %.3f' % mse)
    print('RMSE = %.3f' % rmse)

    plt.rcParams['figure.figsize'] = [15, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    plt.plot(pva['actual'], pva['predicted'], 'o')
    # ax = plt.axes()
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title('EXP:')  # TODO: Update naming scheme
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]
    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)
    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    # plt.savefig(model_name+'-' +'.png')
    # plt.show()
    return plt  # Can I store a graph as an attribute to a model?



# This function evaluates the classification model's performance on the test data using multiple different metrics.
def classification_metrics(prediction, y_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score


    accuracy = accuracy_score(prediction, y_test)
    confusion = confusion_matrix(prediction, y_test)
    classification = classification_report(prediction, y_test)
    roc_auc_score = roc_auc_score(prediction, y_test)

    return accuracy, confusion, classification, roc_auc_score