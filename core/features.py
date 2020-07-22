from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import time, sleep
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from rdkit import Chem

from core import MLMySqlConn


def __get_feat_meth__(model):
    feat_meth = model.feat_meth

    # available featurization options
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    if feat_meth is None:  # ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        feat_meth = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
    selected_feat = [feat_sets[i] for i in feat_meth]
    return selected_feat


def featurize_from_mysql(model):
    model.feat_method_name = __get_feat_meth__(model)

    print("Pulling data from MySql")
    if model.mysql_params is None:
        raise Exception("No connection to MySql made. Please run [model].connect_mysql(**params)")
    start_feat = time()
    mysql_conn = MLMySqlConn(user=model.mysql_params['user'], password=model.mysql_params['password'],
                             host=model.mysql_params['host'], database=model.mysql_params['database'])
    model.data = mysql_conn.retrieve_data(dataset=model.dataset, feat_meth=model.feat_meth)
    model.feat_time = time() - start_feat
    return model.data, model.feat_time, model.feat_method_name


def featurize(model, not_silent=True):
    """
    Caclulate molecular features.
    Returns DataFrame, list of selected features (numeric values. i.e [0,4]),
     and time to featurize.
    Keyword arguments:
    feat_meth -- Features you want by their numerical value.  Default = None (require user input)
    """

    selected_feat = __get_feat_meth__(model)
    model.feat_method_name = selected_feat
    df = model.data

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
    model.data = df
    model.feat_time = feat_time

    return model.data, model.feat_time, model.feat_method_name


def data_split(model, test=0.2, val=0, random=None):
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
    molecules_array = np.array(model.data['smiles'])  # Grab molecules array
    model.test_percent = test
    model.val_percent = val
    model.train_percent = 1 - test - val

    # remove targets from features
    # axis 1 is the columns.
    if model.dataset in ['sider.csv', 'clintox.csv']:
        model.target_name.extend(["smiles"])
        features = model.data.drop(model.target_name, axis=1)

    else:
        features = model.data.drop([model.target_name, 'smiles'], axis=1)

    # save list of strings of features
    model.feature_list = list(features.columns)
    model.feature_length = len(model.feature_list)
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
    model.train_features, model.test_features, model.train_target, model.test_target = train_test_split(
        model.feature_array,
        model.target_array,
        test_size=model.test_percent,
        random_state=model.random_seed)
    # Define smiles that go with the different sets
    model.train_molecules, model.test_molecules, temp_train_target, temp_test_target = train_test_split(
        molecules_array,
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
        b = val / (1 - test)
        model.train_features, model.val_features, model.train_target, model.val_target = train_test_split(
            model.train_features,
            model.train_target,
            test_size=b,
            random_state=model.random_seed)
        # Define smiles that go with the different sets
        # Use temp dummy variables for splitting molecules up the same way
        temp_train_molecules, model.val_molecules, temp_train_target, temp_val_target = train_test_split(
            model.train_molecules,
            temp_train_target,
            test_size=b,
            random_state=model.random_seed)
        # scale the validation features too
        model.val_features = scaler.transform(model.val_features)

        model.n_val = model.val_features.shape[0]
        pval = model.n_val / model.n_tot * 100

    else:
        model.val_features = None
        model.val_target = None
        model.val_molecules = None
        model.n_val = None
        pval = 0

    model.n_train = model.train_features.shape[0]
    ptrain = model.n_train / model.n_tot * 100

    model.n_test = model.test_features.shape[0]
    ptest = model.n_test / model.n_tot * 100

    print()
    print('Dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
        model.n_tot, ptrain, pval, ptest))

    # Logic to seperate data in test/train/val
    def __fetch_set__(smiles):
        if smiles in model.test_molecules:
            return 'test'
        elif smiles in model.train_molecules:
            return 'train'
        else:
            return 'val'

    model.data['in_set'] = model.data['smiles'].apply(__fetch_set__)
    cols = list(model.data.columns)
    cols.remove('in_set')
    model.data = model.data[['in_set', *cols]]

    return (model.data, model.scaler, model.in_shape, model.n_tot,
            model.n_train, model.n_test, model.n_val, model.target_name, model.target_array,
            model.feature_array, model.feature_list, model.feature_length,
            model.train_features, model.test_features, model.val_features,
            model.train_percent, model.test_percent, model.val_percent,
            model.train_target, model.test_target, model.val_target,
            model.train_molecules, model.test_molecules, model.val_molecules)

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
