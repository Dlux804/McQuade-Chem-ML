'''
Goal is to create a way to automate testing multiple models with multiple feature sets
on multiple data sets.  Idea is to churn out many results with one script.

Aimed at 3 datasets (logP, water solubility, hydration energy).
Will use Descriptastorus for features.
Models will be GDB, RF and SVM.  GDB will be test case.
-- Adam Luxon 10/31/2019
'''

import pandas as pd
import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
import matplotlib.pyplot as plt



datasets = ['water-energy.csv', 'Lipophilicity-ID.csv', 'ESOL.csv']

#get the data, locate smiles and exp and drop everything else
def getData(name):
    df = pd.read_csv(name)
    if name == 'Lipophilicity-ID.csv':
        df = df.drop(['CMPD_CHEMBLID', 'ID'], axis = 1)
    elif name == 'ESOL.csv':
        df = df.drop(['Compound ID', 'ESOL predicted log solubility in mols per litre','Minimum Degree', 'Molecular Weight','Number of H-Bond Donors', 'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area'],axis=1)
        df = df.rename(columns= {"water-sol": "exp"})
    else:
        df = df.drop(['iupac', 'calc'], axis=1)
        df = df.rename(columns={'expt': 'exp'})
    return df



# Time to featurize!
def feature_select(df, selected_feat = None):
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts', 'morganchiral3counts', 'atompaircounts']

    if selected_feat == None: #ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
        selected_feat = [int(x) for x in input('Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

    selected_feat = [feat_sets[i] for i in selected_feat]
    print("You have selected the following featurizations: ", end="   ", flush=True)
    print(*selected_feat, sep=', ')


    # This specifies what features to calculate
    generator = MakeGenerator(selected_feat)
    columns = []

    # get the names of the features for column labels
    for name, numpy_type in generator.GetColumns():
        columns.append(name)
    smi = df['smiles']
    data = []
    print('Calculating features...',end=' ', flush=True)
    for mol in smi:
        # actually calculate the descriptors.  Function accepts a smiles
        desc = generator.process(mol)
        data.append(desc)
    print('Done.')

    # make dataframe of all features and merge with lipo df
    features = pd.DataFrame(data, columns=columns)
    df = pd.concat([df, features], axis=1)
    df = df.dropna()
    return df

# df = feature_select(df) # it will ask for





def targets_features(df, random = None):
    '''Take in a data frame, the target column name, and list of columns to be dropped
    returns a numpy array with the target variable,
    a numpy array (matrix) of feature variables,
    and a list of strings of the feature headers.'''

    # make array of target values
    target = np.array(df['exp'])

    # remove target from features
    # axis 1 is the columns.
    features = df.drop(['exp', 'smiles'], axis=1)

    # save list of strings of features
    feature_list = list(features.columns)

    # convert features to numpy
    featuresarr = np.array(features)

    train_percent = 0.8
    test_percent = 1 - train_percent
    train_features, test_features, train_target, test_target = train_test_split(featuresarr, target,
                                                                                test_size=test_percent,
                                                                                random_state=random)  # what data to split and how to do it.
    # print('Total Feature Shape:', features.shape)
    # print('Total Target Shape', target.shape)
    # print()
    # print('Training Features Shape:', train_features.shape)
    # print('Training Target Shape:', train_target.shape)
    # print()
    # print('Test Features Shape:', test_features.shape)
    # print('Test Target Shape:', test_target.shape)
    # print()
    #
    # print('Train:Test -->', np.round(train_features.shape[0] / features.shape[0] * 100, -1), ':',
    #       np.round(test_features.shape[0] / features.shape[0] * 100, -1))

    return train_features, test_features, train_target, test_target, feature_list


# split the data into training and test sets


def gdb_tune(params = None):
    default = {'n_estimators': 1250, 'min_samples_split': 2, 'min_samples_leaf': 26, 'max_features': 'sqrt', 'max_depth': None, 'learning_rate': 0.05}
    if params == None: # do hyper tuning
        # First create the base model to tune
        gdb = GradientBoostingRegressor()

        # make parameter grid
        n_estimators = [int(x) for x in np.linspace(start=500, stop=2000, num=15)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 25, num=5, endpoint=True)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [int(x) for x in np.linspace(2, 30, num=10, endpoint=True)]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(2, 30, num=10, endpoint=True)]
        # learning rate
        learning_rate = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'learning_rate': learning_rate}
        # Fit the random search model

        gdb_random = RandomizedSearchCV(estimator=gdb, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                        random_state=42, n_jobs=2)

        gdb_random.fit(train_features, train_target)
        tuned = gdb_random.best_params_
        return tuned
    else:
        return params


def gdb_train_predict(params, train_features, test_features, train_target, test_target, title, features):

    # set up the model
    gdb = GradientBoostingRegressor(n_estimators=params['n_estimators'], max_features=params['max_features'],
                               max_depth=params['max_depth'], min_samples_split=params['min_samples_split']
                               , min_samples_leaf=params['min_samples_leaf'], learning_rate = params['learning_rate'],
                               random_state=25)

    # train the model!!!!
    gdb.fit(train_features, train_target)

    # predictions
    predictions = gdb.predict(test_features)
    true = test_target
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = true
    pva['predicted'] = predictions
    # print(pva)
    pva.to_csv(title + str(features) + '-pva_data.csv')

    return pva


def pva_graph(pva, title, feat):
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
    print('R^2 = %.3f' % r2)
    print('MSE = %.3f' % mse)
    print('RMSE = %.3f' % rmse)

    plt.rcParams['figure.figsize'] = [9, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    plt.plot(pva['actual'], pva['predicted'], 'o')
    # ax = plt.axes()
    plt.xlabel('True '+title, fontsize=18);
    plt.ylabel('Predicted '+title, fontsize=18);
    # plt.title('EXP:'+exp+'-'+graph_title)
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

    plt.savefig(title + str(feat) + '-PvA.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()

    return r2, mse, rmse


def replicate_model(df, params, n, title, feat):
    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    for i in range(0,n): # run model n times
        train_features, test_features, train_target, test_target, feature_list = targets_features(df, random=None)
        pva = gdb_train_predict(params, train_features, test_features, train_target,
                                            test_target,
                                            title, feat)

        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
        # print('r2=', r2)

    r2_avg = r2.mean()
    r2_std = r2.std()
    mse_avg = mse.mean()
    mse_std = mse.std()
    rmse_avg = rmse.mean()
    rmse_std = rmse.std()
    print('Average R^2 = %.3f' % r2_avg, '+- %.3f' % r2_std)
    print('Average RMSE = %.3f' % rmse_avg, '+- %.3f' % rmse_std)
    print()
    return r2_avg, r2_std, mse_avg, mse_std, rmse_avg, rmse_std



def run_models(model,  tuning = False):
    results = pd.DataFrame([], columns=['Data Set', 'Features', 'Hyperparamters', 'r2_avg', 'r2_std', 'mse_avg', 'mse_std', 'rmse_avg', 'rmse_std'])
    if model == 'gdb':
        for data in datasets:
            if data == 'Lipophilicity-ID.csv':
                title = 'logP'
            if data == 'ESOL.csv':
                title = 'log water solubility'
            if data ==  'water-energy.csv':
                title = 'Hydration Energy'

            for feat in feature_array:
                df = getData(data)
                df = feature_select(df,feat)
                train_features, test_features, train_target, test_target, feature_list = targets_features(df, random=42)
                gdb_params = gdb_tune({'n_estimators': 1250, 'min_samples_split': 2, 'min_samples_leaf': 26, 'max_features': 'sqrt', 'max_depth': None, 'learning_rate': 0.05})

                gdb_predictions = gdb_train_predict(gdb_params, train_features, test_features, train_target, test_target,
                                                    title, feat)
                r2, mse, rmse = pva_graph(gdb_predictions, title, feat)
                r2_avg, r2_std, mse_avg, mse_std, rmse_avg, rmse_std = replicate_model(df,gdb_params, 5, title, feat)
                outcome = [title, feat, gdb_params, r2_avg, r2_std, mse_avg, mse_std, rmse_avg, rmse_std]
                results.loc[len(results)] = outcome
                # print(results)
    else:
        pass
    results.to_csv('HTE-Model-results.csv')
    return results

feature_array = [[0], [0,2], [0,3], [0,4], [0,6], [2], [3], [4], [6]]

# exp = 'XXX'
# df = getData(datasets[2])
# df = feature_select(df)
# print(df)
# print()
# train_features, test_features, train_target, test_target, feature_list  = targets_features(df)
# gdb_params = gdb_tune({'n_estimators': 1250, 'min_samples_split': 2, 'min_samples_leaf': 26, 'max_features': 'sqrt', 'max_depth': None, 'learning_rate': 0.05})
# print(gdb_params)
# gdb_predictions = gdb_train_predict(gdb_params, train_features, test_features, train_target, test_target, 'Test', [0,2])
# r2, mse, rmse = pva_graph(gdb_predictions)

final = run_models('gdb')