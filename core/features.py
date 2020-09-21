from time import time, sleep

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def canonical_smiles(df):
    """
    Objective: Create list of canonical SMILES from SMILES
    Intent: While the SMILES in dataset from moleculenet.ai are all canonical, it is always good to be safe. I don't
            know if I should add in a way to detect irregular SMILES and remove the rows that contains them in the
            dataframe. However, that process should be carried out at the start of the pipeline instead of at the end.
    :param smiles_list:
    :return:
    """
    smiles = df['smiles']
    con_smiles = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            con_smiles.append(Chem.MolToSmiles(mol))
        else:
            con_smiles.append('bad_smiles')
    df['smiles'] = con_smiles
    df = df.loc[df['smiles'] != 'bad_smiles']
    return df


def featurize(self, not_silent=True, retrieve_from_mysql=False):
    """
    Caclulate molecular features.
    Returns DataFrame, list of selected features (numeric values. i.e [0,4]),
     and time to featurize.
    Keyword arguments:
    feat_meth -- Features you want by their numerical value.  Default = None (require user input)
    """
    feat_meth = self.feat_meth
    df = self.data
    df = canonical_smiles(df=df)  # Turn SMILES into CANONICAL SMILES
    # available featurization options
    feat_sets = ['rdkit2d', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts', 'morganchiral3counts',
                 'atompaircounts']

    if feat_meth is None:  # ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        feat_meth = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
    selected_feat = [feat_sets[i] for i in feat_meth]

    self.selected_feat_string = '-'.join(selected_feat) # This variable will be used later in train.py for giving classification roc graph a unique file name.

    self.feat_method_name = selected_feat

    # Get data from MySql if called
    if retrieve_from_mysql:
        print("Pulling data from MySql")
        self.featurize_from_mysql()
        return

    if not_silent:  # Add option to silence messages
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

    # The following section removes rows that had failed featurizations. This makes the workflow run properly for
    # both the clintox and the BBBP data sets.

    issue_row_list = []
    issue_row = 0
    for smiles in smi:
        x = Chem.MolFromSmiles(smiles)
        if x == None:
            issue_row_list.append(issue_row)
        issue_row = issue_row + 1

    rows = df.index[[issue_row_list]]
    df.drop(rows, inplace=True)
    smi.drop(rows, inplace=True)

    smi2 = tqdm(smi, desc="Featurization")  # for progress bar
    data = list(map(generator.process, smi2))
    if not_silent:
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
    if self.dataset in ['sider.csv', 'clintox.csv']:
        self.target_name.extend(["smiles"])
        features = self.data.drop(self.target_name, axis=1)

    else:
        features = self.data.drop([self.target_name, 'smiles'], axis=1)
    # save list of strings of features
    self.feature_list = list(features.columns)
    self.feature_length = len(self.feature_list)
    # convert features to numpy
    featuresarr = np.array(features)
    # n_total = featuresarr.shape[0]

    # store to instance
    self.feature_array = featuresarr
    self.n_tot = self.feature_array.shape[0]
    self.in_shape = self.feature_array.shape[1]
    if self.algorithm == 'cnn':
        self.feature_array = self.feature_array.reshape(self.feature_array.shape[0], self.feature_array.shape[1], 1)

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
    # self.train_features = scaler.fit_transform(self.train_features)
    # self.test_features = scaler.transform(self.test_features)

    # Can scale data 1d, 2d and 3d data
    self.train_features = scaler.fit_transform(self.train_features.reshape(-1,
                                               self.train_features.shape[-1])).reshape(self.train_features.shape)
    self.test_features = scaler.transform(self.test_features.reshape(-1,
                                               self.test_features.shape[-1])).reshape(self.test_features.shape)

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
        self.train_molecules, self.val_molecules, temp_train_target, temp_val_target = train_test_split(
            self.train_molecules,
            temp_train_target,
            test_size=b,
            random_state=self.random_seed)
        # scale the validation features too
        # self.val_features = scaler.transform(self.val_features)
        self.val_features = scaler.transform(self.val_features.reshape(-1,
                                             self.val_features.shape[-1])).reshape(self.val_features.shape)
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
    # print('Total Target Shape', self.target_array.shape)
    # print()
    # print('Training Features Shape:', self.train_features.shape)
    # print('Training Target Shape:', self.train_target.shape)
    # print()
    # print('Test Features Shape:', self.test_features.shape)
    # print('Test Target Shape:', self.test_target.shape)
    # print()
    #
    # print('Train:Test -->', np.round(self.train_features.shape[0] / features.shape[0] * 100, -1), ':',
    #       np.round(self.test_features.shape[0] / features.shape[0] * 100, -1))

    # return train_features, test_features, train_target, test_target, feature_list