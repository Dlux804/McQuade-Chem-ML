from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import time, sleep
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def featurize(model):
    """
    Caclulate molecular features.
    Returns DataFrame, list of selected features (numeric values. i.e [0,4]),
     and time to featurize.

    Keyword arguments:
    feat_meth -- Features you want by their numerical value.  Default = None (require user input)
    """
    feat_meth = model.feat_meth
    df = model.data

    # available featurization options
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    if feat_meth is None:  # ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        feat_meth = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

    print(feat_sets, feat_meth)

    selected_feat = [feat_sets[i] for i in feat_meth]
    print("You have selected the following featurizations: ", end="   ", flush=True)
    print(*selected_feat, sep=', ')
    print('Calculating features...')
    sleep(0.25)

    # Start timer
    start_feat = time()

    # Use descriptastorus generator
    generator = MakeGenerator(selected_feat)
    columns = []

    # get the names of the features for column labels
    for name, numpy_type in generator.GetColumns():
        columns.append(name)
    smi = df['smiles']

    smi2 = tqdm(smi, desc= "Featurizaiton")  # for progress bar
    data = list(map(generator.process, smi2))
    print('Done.')
    stop_feat = time()
    feat_time = stop_feat - start_feat

    # make dataframe of all features
    features = pd.DataFrame(data, columns=columns)
    df = pd.concat([df, features], axis=1)
    df = df.dropna()

    # remove the "RDKit2d_calculated = True" column(s)
    df = df.drop(list(df.filter(regex='_calculated')), axis=1)
    df = df.drop(list(df.filter(regex='[lL]og[pP]')), axis=1)

    # store data back into the instance
    model.data = df
    model.feat_timedf = feat_time
    return model

def data_split(model, test=0.2, val=0, random = None):
    """
    Take in a data frame, the target column name (exp).
    Returns a numpy array with the target variable,
    a numpy array (matrix) of feature variables,
    and a list of strings of the feature headers.

    Keyword Arguments
    random -- Integer. Set random seed using in data splitting.  Default = None
    test -- Float(0.0-1.0).  Percent of data to be used for testing
    val -- Float.  Percent of data to be used for validation.  Taken out of training data after test.
    """

    # make array of target values
    model.target_array = np.array(model.data[model.target_name])
    model.test_percent = test
    model.val_percent = val
    model.train_percent = 1 - test - val

    # remove target from features
    # axis 1 is the columns.
    # features = df.drop([exp, 'smiles'], axis=1)
    # TODO Collect and store the molecules in train, test and validation data sets
    features = model.data.drop([model.target_name, 'smiles'], axis=1)

    # save list of strings of features
    model.feature_list = list(features.columns)

    # convert features to numpy
    featuresarr = np.array(features)
    # n_total = featuresarr.shape[0]

    # store to instance
    model.feature_array = featuresarr
    model.n_tot = model.feature_array.shape[0]
    model.in_shape = model.feature_array.shape[1]

    # print("self.feature_array: ", self.feature_array)
    # print('self target array', self.target_array)
    # print('Total counts:', self.n_tot)
    # print('Feature input shape', self.in_shape)

    # what data to split and how to do it.
    model.train_features, model.test_features, model.train_target, model.test_target = train_test_split(model.feature_array,
                                                                                                        model.target_array,
                                                                                                        test_size=model.test_percent,
                                                                                                        random_state=model.random_seed)
    # scale the data.  This should not hurt but can help many models
    # TODO add this an optional feature
    # TODO add other scalers from sklearn
    scaler = StandardScaler()
    model.scaler = scaler
    model.train_features = scaler.fit_transform(model.train_features)
    model.test_features = scaler.transform(model.test_features)

    if val > 0:  # if validation data is requested.
        # calculate percent of training to convert to val
        b = val / (1-test)
        model.train_features, model.val_features, model.train_target, model.val_target = train_test_split(model.train_features,
                                                                                                          model.train_target, test_size=b,
                                                                                                          random_state=model.random_seed)
        # scale the validation features too
        model.val_features = scaler.transform(model.val_features)

        n_total = model.n_tot

        model.n_train = model.train_features.shape[0]
        ptrain = model.n_train / n_total * 100


        model.n_test = model.test_features.shape[0]
        ptest = model.n_test / n_total * 100

        model.n_val = model.val_features.shape[0]
        pval = model.n_val / n_total * 100

        print('The dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
                n_total, ptrain, pval, ptest))

        # return train_features, test_features, val_features, train_target, test_target, val_target, feature_list

    # Uncomment this section to have data shape distribution printed.

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

    # return train_features, test_features, train_target, test_target, feature_list

    return model
