import random
from math import floor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from rdkit import Chem
import pandas as pd


def __dropCol__(df, target_name, keepSmiles=False):
    """"""
    if keepSmiles:
        target_name.extend(["smiles"])
        return df.drop(target_name, axis=1)
    else:
        return df.drop([target_name, 'smiles'], axis=1)


def data_split(self, test=0.2, val=0, add_molecules_to_testset=None):
    feature_columns = list(self.data.columns)
    feature_columns.remove('smiles')
    feature_columns.remove(self.target_name)
    self.feature_list = feature_columns
    self.feature_length = len(feature_columns)

    sorted_data = []
    for index, row in self.data.iterrows():
        row = dict(row)
        row_data = {'smiles': row['smiles'], 'target': row[self.target_name]}
        features = []
        for label, value in row.items():
            if label != 'smiles' and label != self.target_name:
                features.append(value)
        row_data['features'] = features
        sorted_data.append(row_data)

    random.seed(self.random_seed)
    random.shuffle(sorted_data)

    sorted_data_df = pd.DataFrame(sorted_data)
    self.feature_array = np.array([np.array(x) for x in sorted_data_df['features']])
    if self.algorithm == 'cnn':
        self.feature_array = self.feature_array.reshape(self.feature_array.shape[0], self.feature_array.shape[1], 1)
    self.in_shape = self.feature_array.shape[1]
    self.target_array = np.array(sorted_data_df['target'])

    self.n_tot = len(sorted_data)
    n_val = floor(self.n_tot * val)
    n_test = floor(self.n_tot * test)
    n_train = self.n_tot - n_val - n_test
    self.val_percent = n_val / self.n_tot * 100
    self.test_percent = n_test / self.n_tot * 100
    self.train_percent = n_train / self.n_tot * 100

    val_split_point = n_train
    test_split_point = n_train + n_val

    train = sorted_data[:val_split_point]
    val = sorted_data[val_split_point:test_split_point]
    test = sorted_data[test_split_point:]

    if add_molecules_to_testset is not None:

        def __msmtts__(a, b):  # move specified molecules to test set
            i = 0
            while i < len(b):
                if b[i]['smiles'] in add_molecules_to_testset:
                    a.append(b[i])
                    b.pop(i)
                    i = i - 1  # size of list has decreased by one
                i = i + 1

        __msmtts__(test, train)
        __msmtts__(test, val)

    def __gdfdl__(scaler, a):  # gather data from data list
        df = pd.DataFrame(a)
        molecules_array = np.array(df['smiles'].tolist())
        target_array = np.array(df['target'].tolist())
        features_array = df['features'].tolist()
        features_array = np.array([np.array(x) for x in features_array])
        features_array = scaler.fit_transform(features_array.reshape(-1, features_array.shape[-1])).reshape(
            features_array.shape)
        return features_array, target_array, molecules_array

    scaler = StandardScaler()
    self.scaler = scaler
    self.train_features, self.train_target, self.train_molecules = __gdfdl__(scaler, train)
    self.val_features, self.val_target, self.val_molecules = __gdfdl__(scaler, val)
    self.test_features, self.test_target, self.test_molecules = __gdfdl__(scaler, test)

    print()
    print(
        'Dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
            self.n_tot, self.train_percent, self.val_percent, self.test_percent))

    self.in_shape = self.feature_array.shape[1]

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

    print('Total Feature Shape:', self.feature_array.shape)
    print('Total Target Shape', self.target_array.shape)
    print()
    print('Training Features Shape:', self.train_features.shape)
    print('Training Target Shape:', self.train_target.shape)
    print()
    print('Test Features Shape:', self.test_features.shape)
    print('Test Target Shape:', self.test_target.shape)
    print()

    print('Train:Test -->', np.round(self.train_features.shape[0] / self.feature_array.shape[0] * 100, 1), ':',
          np.round(self.test_features.shape[0] / self.feature_array.shape[0] * 100, 1))
