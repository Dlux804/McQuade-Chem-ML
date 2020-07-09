from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import time, sleep
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def featurize(self):
    """
    Caclulate molecular features.
    Returns DataFrame, list of selected features (numeric values. i.e [0,4]),
     and time to featurize.

    Keyword arguments:
    feat_meth -- Features you want by their numerical value.  Default = None (require user input)
    """
    feat_meth = self.feat_meth
    df = self.data

    # available featurization options
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    if feat_meth is None:  # ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        feat_meth = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
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

    smi2 = tqdm(smi, desc= "Featurization")  # for progress bar
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
    self.data = df
    self.feat_time = feat_time


def data_split(self, test=0.2, val=0, random=None):
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
    self.target_array = np.array(self.data[self.target_name])
    molecules_array = np.array(self.data['smiles'])  # Grab molecules array
    self.test_percent = test
    self.val_percent = val
    self.train_percent = 1 - test - val

    # remove targets from features
    # axis 1 is the columns.
    if self.task_type == 'classification':
        self.target_name.extend(["smiles"])
        features = self.data.drop(self.target_name, axis=1)

    if self.task_type == 'regression':
        features = self.data.drop([self.target_name, 'smiles'], axis=1)



    # save list of strings of features
    self.feature_list = list(features.columns)

    # convert features to numpy
    featuresarr = np.array(features)
    # n_total = featuresarr.shape[0]

    # store to instance
    self.feature_array = featuresarr
    self.n_tot = self.feature_array.shape[0]
    self.in_shape = self.feature_array.shape[1]

    # print("self.feature_array: ", self.feature_array)
    # print('self target array', self.target_array)
    # print('Total counts:', self.n_tot)
    # print('Feature input shape', self.in_shape)

    # what data to split and how to do it.
    self.train_features, self.test_features, self.train_target, self.test_target = train_test_split(
                                                                                                    self.feature_array,
                                                                                                    self.target_array,
                                                                                                    test_size=self.test_percent,
                                                                                                    random_state=self.random_seed)
    self.raw_train_features = self.train_features
    # Define smiles that go with the different sets
    self.train_molecules, self.test_molecules, temp_train_target, temp_test_target = train_test_split(
                                                                                                    molecules_array,
                                                                                                    self.target_array,
                                                                                                    test_size=self.test_percent,
                                                                                                    random_state=self.random_seed)

    # scale the data.  This should not hurt but can help many models
    # TODO add this an optional feature
    # TODO add other scalers from sklearn
    scaler = StandardScaler()
    self.scaler = scaler
    self.train_features = scaler.fit_transform(self.train_features)
    self.test_features = scaler.transform(self.test_features)

    if val > 0:  # if validation data is requested.
        # calculate percent of training to convert to val
        b = val / (1 - test)
        self.train_features, self.val_features, self.train_target, self.val_target = train_test_split(
                                                                                                    self.train_features,
                                                                                                    self.train_target,
                                                                                                    test_size=b,
                                                                                                    random_state=self.random_seed)
        # Define smiles that go with the different sets
        # Use temp dummy variables for splitting molecules up the same way
        temp_train_molecules, self.val_molecules, temp_train_target, temp_val_target = train_test_split(
                                                                                                    self.train_molecules,
                                                                                                    temp_train_target,
                                                                                                    test_size=b,
                                                                                                    random_state=self.random_seed)
        # scale the validation features too
        self.val_features = scaler.transform(self.val_features)

        self.n_val = self.val_features.shape[0]
        pval = self.n_val / self.n_tot * 100

    else:
        pval = 0

    self.n_train = self.train_features.shape[0]
    ptrain = self.n_train / self.n_tot * 100

    self.n_test = self.test_features.shape[0]
    ptest = self.n_test / self.n_tot * 100

    print()
    print('Dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
            self.n_tot, ptrain, pval, ptest))

    # Logic to seperate data in test/train/val
    def __fetch_set__(smiles):
        if smiles in self.test_molecules:
            return 'test'
        elif smiles in self.train_molecules:
            return 'train'
        else:
            return 'val'

    self.data['in_set'] = self.data['smiles'].apply(__fetch_set__)
    cols = list(self.data.columns)
    cols.remove('in_set')
    self.data = self.data[['in_set', *cols]]

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
