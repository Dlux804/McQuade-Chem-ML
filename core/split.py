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


def data_split(self, test=0.2, givenSmiles=None, dropSmiles=None, val=0, scaler=None, random=None):
    """
    Take in a data frame, the target column name (exp).
    Returns a numpy array with the target variable,
    a numpy array (matrix) of feature variables,
    and a list of strings of the feature headers.

    :param self:
    :param test:
    :param val:
    :param scaler:
    :param random:
    :return:
    """
    self.test_percent = test
    self.val_percent = val
    self.target_array = np.array(self.data[self.target_name])

    if scaler == "std" or scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()

    self.scaler = scaler

    if self.dataset in ['sider.csv', 'clintox.csv']:
        features_df = __dropCol__(df=self.data, target_name=self.target_name, keepSmiles=True)
    else:
        features_df = __dropCol__(df=self.data, target_name=self.target_name)
    if dropSmiles is not None:
        dropSmiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in dropSmiles]
        features_df = features_df[~features_df.isin(dropSmiles)]

    self.feature_list = list(features_df.columns)  # save list of strings of features
    self.feature_length = len(self.feature_list)  # Save feature size
    self.feature_array = np.array(features_df)
    self.n_tot = self.feature_array.shape[0]
    temp_test_size = 0
    pval = 0

    if isinstance(test, list):  #
        try:
            test = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in test]
        except Exception:
            raise Exception(
                "{} is not a valid input".format(test)
            )

        if test is not None:
            test_data = pd.concat(self.data[self.data['smiles'] == i] for i in test)
        else:
            raise Exception(
                "{} is not a valid SMILES. Please try again with a different SMILES".format(test)
            )

        self.test_target = np.array(test_data[self.target_name])  # Array of test target
        self.test_molecules = np.array(test_data['smiles'])  # Test molecules
        self.test_features = np.array(
            __dropCol__(df=test_data, target_name=self.target_name))  # Test feature array
        self.test_percent = self.test_features.shape[0] / len(self.data)  # test percent

        train_data = self.data[~self.data['smiles'].isin(test)]  # Dataframe of train Feature
        self.train_molecules = np.array(train_data['smiles'])  # Train Molecules
        self.train_target = np.array(train_data[self.target_name])  # array of train target
        self.train_features = np.array(__dropCol__(train_data, self.target_name))  # Array of train features

        self.train_percent = 1 - self.test_percent - self.val_percent  # train_percent
        temp_test_size = self.val_percent  # temp_test_size to split train to train and val

        if val > 0:  # if validation data is requested.
            # calculate percent of training to convert to val

            self.train_features, self.val_features, self.train_target, self.val_target = train_test_split(
                self.train_features,
                self.train_target,
                test_size=self.val_percent,
                random_state=self.random_seed)
            # Define smiles that go with the different sets
            # Use temp dummy variables for splitting molecules up the same way
            self.train_molecules, self.val_molecules, temp_train_target, temp_val_target = train_test_split(
                self.train_molecules,
                self.train_target,
                test_size=self.val_percent,
                random_state=self.random_seed)

    elif isinstance(test, float):
        molecules_array = np.array(self.data['smiles'])  # Grab molecules array
        self.train_percent = 1 - self.test_percent - self.val_percent
        feature_array = self.feature_array
        target_array = self.target_array
        # add_data_molecules = np.array()
        # add_features_array = np.array()
        # add_target_array = np.array()
        if givenSmiles is not None:
            givenSmiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in givenSmiles]
            add_data = self.data[self.data['smiles'].isin(givenSmiles)]
            add_data_molecules = np.array(add_data['smiles'])
            add_features_array = np.array(__dropCol__(add_data, self.target_name))
            add_target_array = np.array(add_data[self.target_name])

            features_df = self.data[~self.data['smiles'].isin(givenSmiles)]
            feature_array = np.array(__dropCol__(features_df, self.target_name))
            target_array = np.array(features_df[self.target_name])
            molecules_array = np.array(features_df['smiles'])

        self.train_features, self.test_features, self.train_target, self.test_target = train_test_split(
            feature_array,
            target_array,
            test_size=self.test_percent,
            random_state=self.random_seed)
        # Define smiles that go with the different sets
        self.train_molecules, self.test_molecules, temp_train_target, temp_test_target = train_test_split(
            molecules_array,
            target_array,
            test_size=self.test_percent,
            random_state=self.random_seed)
        if givenSmiles is not None:
            self.test_features = np.concatenate([self.test_features, add_features_array])
            self.test_molecules = np.concatenate([self.test_molecules, add_data_molecules])
            self.test_target = np.concatenate([self.test_target, add_target_array])

        temp_test_size = val / (1 - test)

    if val > 0:  # if validation data is requested.
        # calculate percent of training to convert to val
        self.train_features, self.val_features, self.train_target, self.val_target = train_test_split(
            self.train_features,
            self.train_target,
            test_size=temp_test_size,
            random_state=self.random_seed)

        # Define smiles that go with the different sets
        # Use temp dummy variables for splitting molecules up the same way
        # self.train_molecules, self.val_molecules, temp_train_target, temp_val_target = train_test_split(
        #     self.train_molecules,
        #     self.train_target,
        #     test_size=temp_test_size,
        #     random_state=self.random_seed)
        self.n_val = self.val_features.shape[0]
        pval = self.n_val / self.n_tot * 100

    if self.algorithm != "cnn" and self.scaler is not None:
        self.train_features = self.scaler.fit_transform(self.train_features)
        self.test_features = self.scaler.transform(self.test_features)
        if val > 0:
            self.val_features = self.scaler.transform(self.val_features)
    elif self.algorithm == "cnn" and self.scaler is not None:
        # Can scale data 1d, 2d and 3d data
        self.train_features = self.scaler.fit_transform(self.train_features.reshape(-1,
                                                        self.train_features.shape[-1])).reshape(self.train_features.shape)

        self.test_features = self.scaler.transform(self.test_features.reshape(-1,
                                                   self.test_features.shape[-1])).reshape(self.test_features.shape)
        if val > 0:
            self.val_features = self.scaler.transform(self.val_features.reshape(-1,
                                                      self.val_features.shape[-1])).reshape(self.val_features.shape)
    else:
        pass
    self.n_train = self.train_features.shape[0]
    ptrain = self.n_train / self.n_tot * 100

    self.n_test = self.test_features.shape[0]
    ptest = self.n_test / self.n_tot * 100

    print()
    print(
        'Dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
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