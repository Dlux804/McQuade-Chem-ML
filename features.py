from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def featurize(df, model_name, num_feat=None):
    """ Caclulate molecular features.  Returns DataFrame and list of selected features (numeric values. i.e [0,4]).

    Keyword arguments:
    num_feat -- Features you want by their numerical value.  Default = None (require user input)
    """

    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    # Remove un-normalized feature option depending on model type
    if model_name == 'nn' or model_name == 'knn':
        feat_sets.remove('rdkit2d')
        print(feat_sets)
        if num_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            num_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

        selected_feat = [feat_sets[i] for i in num_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')

    # un-normalized features are OK
    else:
        if num_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            num_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
        selected_feat = [feat_sets[i] for i in num_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')

    # Use descriptastorus generator
    generator = MakeGenerator(selected_feat)
    columns = []

    # get the names of the features for column labels
    for name, numpy_type in generator.GetColumns():
        columns.append(name)
    smi = df['smiles']
    data = []
    print('Calculating features...', end=' ', flush=True)
    for mol in smi:
        # actually calculate the descriptors.  Function accepts a smiles
        desc = generator.process(mol)
        data.append(desc)
    print('Done.')

    # make dataframe of all features and merge with lipo df
    features = pd.DataFrame(data, columns=columns)
    df = pd.concat([df, features], axis=1)
    df = df.dropna()

    return df, num_feat


def targets_features(df, exp, train=0.8, random = None):
    """Take in a data frame, the target column name (exp).
    Returns a numpy array with the target variable,
    a numpy array (matrix) of feature variables,
    and a list of strings of the feature headers.

    Keyword Arguments
    random -- Integer. Set random seed using in data splitting.  Default = None"""

    # make array of target values
    target = np.array(df[exp])  # exp input should be target variable string

    # remove target from features
    # axis 1 is the columns.
    features = df.drop([exp, 'smiles'], axis=1)

    # save list of strings of features
    feature_list = list(features.columns)

    # convert features to numpy
    featuresarr = np.array(features)

    train_percent = train
    test_percent = 1 - train_percent
    train_features, test_features, train_target, test_target = train_test_split(featuresarr, target,
                                                                                test_size=test_percent,
                                                                                random_state=random)  # what data to split and how to do it.
    # print('Total Feature Shape:', features.shape)
    # print('Total Target Shape', target.shape)
    # print()
    # # print('Training Features Shape:', train_features.shape)
    # # print('Training Target Shape:', train_target.shape)
    # # print()
    # # print('Test Features Shape:', test_features.shape)
    # # print('Test Target Shape:', test_target.shape)
    # # print()
    #
    # print('Train:Test -->', np.round(train_features.shape[0] / features.shape[0] * 100, -1), ':',
    #       np.round(test_features.shape[0] / features.shape[0] * 100, -1))

    return train_features, test_features, train_target, test_target, feature_list
