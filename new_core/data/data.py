"""
Objective: Split data
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.random import randint
from new_core.data.features import Feature


class Data(Feature):
    def __init__(self, dataset, target, feat_meth, test=0.2, val=0, random=None):
        Feature.__init__(self, dataset, target, feat_meth)
        if random is None:
            self.random_seed = randint(low=1, high=50)
        self.test_percent = test
        self.val_percent = val

    def data_split(self):
        self.train_percent = 1 - self.test_percent - self.val_percent
        self.target_array = np.array(self.data[self.target])
        molecules_array = np.array(self.data['smiles'])  # Grab molecules array

        if self.dataset in ['sider.csv', 'clintox.csv']:
            self.target.extend(["smiles"])
            features = self.data.drop(self.target, axis=1)

        else:
            features = self.data.drop([self.target, 'smiles'], axis=1)

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
        self.train_features = scaler.fit_transform(self.train_features)
        self.test_features = scaler.transform(self.test_features)

        if self.val_percent > 0:  # if validation data is requested.
            # calculate percent of training to convert to val
            b = self.val_percent / (1 - self.test_percent)

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
        print(
            f"""Dataset of {self.n_tot} points is split into training ({ptrain}%), validation ({pval}%), and testing ({ptest}%).""")
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