from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from rdkit import Chem
import pandas as pd
import deepchem as dc
from deepchem.feat import RDKitDescriptors, CircularFingerprint
from collections import OrderedDict

def __dropCol__(df, target_name, keepSmiles=False):
    """"""
    if keepSmiles:
        target_name.extend(["smiles"])
        return df.drop(target_name, axis=1)
    else:
        return df.drop([target_name, 'smiles'], axis=1)


def data_split(self, test=0.2, val=0, split="random", add_molecule_to_testset=None,  scaler=None, random=None):
    """
        Take in a data frame, the target column name (exp).
    Returns a numpy array with the target variable,
    a numpy array (matrix) of feature variables,
    and a list of strings of the feature headers.

    :param self:
    :param test:
    :param val:
    :param split:
            Keywords: 'random', 'index', 'scaffold'. Default is 'random'
    :param add_molecule_to_testset:
    :param scaler:
            Keywords:  None, 'standard', 'minmax'. Default is None
    :param random:
    :return:
    """
    self.test_percent = test  # Test percent instance
    self.val_percent = val  # Val percent instance
    self.target_array = np.array(self.data[self.target_name])  # Target array instance
    scaler_method = scaler
    if scaler == "standard":  # Determine data scaling method
        scaler_method = StandardScaler()
    elif scaler == "minmax":
        scaler_method = MinMaxScaler()

    self.scaler_method = scaler  # Scaler instance

    if self.dataset in ['sider.csv', 'clintox.csv']:  # Drop specific columns for specific datasets
        features_df = __dropCol__(df=self.data, target_name=self.target_name, keepSmiles=True)
    else:
        features_df = __dropCol__(df=self.data, target_name=self.target_name)

    # if dropSmiles is not None:
    #     dropSmiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in dropSmiles]
    #     features_df = features_df[~features_df.isin(dropSmiles)]

    self.feature_list = list(features_df.columns)  # save list of strings of features
    self.feature_length = len(self.feature_list)  # Save feature size
    self.feature_array = np.array(features_df)  # Save feature array instance
    self.n_tot = self.feature_array.shape[0]  # Save total amount of features

    molecules_array = np.array(self.data['smiles'])  # Grab molecules array

    self.train_percent = 1 - self.test_percent - self.val_percent  # Train percent

    temp_data = self.data

    # if val != 0:
    canonMolToAdd = []
    if add_molecule_to_testset is not None:  # For specific molecules to add to testset
        for i in add_molecule_to_testset:
            to_mol = Chem.MolFromSmiles(i)
            if to_mol is not None:
                canonMolToAdd.append(Chem.MolToSmiles(to_mol))
        # add_molecule_to_testset = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in add_molecule_to_testset]

        add_data = self.data[self.data['smiles'].isin(canonMolToAdd)]  # Add in features of specific SMILES

        add_data_molecules = np.array(add_data['smiles'])  # Specific molecule array to be added

        add_features_array = np.array(__dropCol__(add_data, self.target_name))  # Specific feature array to be added

        add_target_array = np.array(add_data[self.target_name])  # Specific target array to be added

        temp_data = self.data[~self.data['smiles'].isin(canonMolToAdd)]  # Final feature df

    # We need to generate fingerprints to use deepchem's scaffold and butina splitting techniques.
    featurizer = CircularFingerprint(size=2048)

    # Loading in csv into deepchem
    loader = dc.data.CSVLoader(tasks=[self.target_name], smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(self.dataset)  # Feature

    split_dict = {"random": dc.splits.RandomSplitter(), "scaffold": dc.splits.ScaffoldSplitter(),
                  "index": dc.splits.IndexSplitter()}  # Dictionary of different data splitting methods
    split_name_dict = {"random": "RandomSplit", "scaffold": "ScaffoldSplit", "index": "IndexSplit"}
    try:
        splitter = split_dict[split]
        self.split_method = split_name_dict[split]
    except KeyError:
        raise Exception("""Invalid splitting methods. Please enter either "random", "scaffold" or "index".""")
    if val == 0:

        train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=round(1-test, 1),
                                                                seed=self.random_seed)
    else:
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=1-test-val,
                                                                                     frac_test=test, frac_valid=val,
                                                                                     seed=self.random_seed)
    # All training related data
    train_molecules = []
    for train_smiles in train_dataset.ids:
        train_to_mol = Chem.MolFromSmiles(train_smiles)
        if train_to_mol is not None:
            train_molecules.append(Chem.MolToSmiles(train_to_mol))
        else:
            pass

    train_molecules = list(OrderedDict.fromkeys(train_molecules))
    train_df = temp_data[temp_data['smiles'].isin(train_molecules)]
    train_df = train_df.drop_duplicates()  # Drop duplicates in Dataframe
    self.train_molecules = np.array(train_df['smiles'])
    self.train_features = np.array(__dropCol__(df=train_df, target_name=self.target_name))
    self.n_train = self.train_features.shape[0]
    self.train_target = np.array(train_df[self.target_name])

    # All testing related data
    test_molecules = []
    for test_smiles in test_dataset.ids:
        test_to_mol = Chem.MolFromSmiles(test_smiles)
        if test_to_mol is not None:
            test_molecules.append(Chem.MolToSmiles(test_to_mol))
        else:
            pass
    test_molecules = list(OrderedDict.fromkeys(test_molecules))  # Drop duplicates in list

    test_df = temp_data[temp_data['smiles'].isin(test_molecules)]
    test_df = test_df.drop_duplicates()  # Drop duplicates in Dataframe
    self.test_molecules = np.array(test_df['smiles'])
    self.test_features = np.array(__dropCol__(df=test_df, target_name=self.target_name))
    self.test_target = np.array(test_df[self.target_name])
    if add_molecule_to_testset is not None:  # If there are specific SMILES to add
        self.test_features = np.concatenate([self.test_features, add_features_array])
        self.test_molecules = np.concatenate([self.test_molecules, add_data_molecules])
        self.test_target = np.concatenate([self.test_target, add_target_array])
    self.n_test = self.test_features.shape[0]

    # All validating related data
    if val != 0:
        val_molecules = []
        for smiles in valid_dataset.ids:
            val_to_mol = Chem.MolFromSmiles(smiles)
            if val_to_mol is not None:
                val_molecules.append(Chem.MolToSmiles(val_to_mol))

        val_molecules = list(OrderedDict.fromkeys(val_molecules))
        val_df = temp_data[temp_data['smiles'].isin(val_molecules)]
        val_df = val_df.drop_duplicates()
        self.val_molecules = np.array(val_df['smiles'])
        self.val_features = np.array(__dropCol__(df=val_df, target_name=self.target_name))
        self.val_target = np.array(val_df[self.target_name])
        self.n_val = self.val_features.shape[0]

    if self.algorithm != "cnn" and scaler is not None:
        self.train_features = scaler_method.fit_transform(self.train_features)
        self.test_features = scaler_method.transform(self.test_features)
        if val > 0:
            self.val_features = scaler_method.transform(self.val_features)
    elif self.algorithm == "cnn" and scaler is not None:
        # Can scale data 1d, 2d and 3d data
        self.train_features = scaler_method.fit_transform(self.train_features.reshape(-1,
                                                                                    self.train_features.shape[
                                                                                        -1])).reshape(
            self.train_features.shape)

        self.test_features = scaler_method.transform(self.test_features.reshape(-1,
                                                                              self.test_features.shape[-1])).reshape(
            self.test_features.shape)
        if val > 0:
            self.val_features = scaler_method.transform(self.val_features.reshape(-1,
                                                                                self.val_features.shape[-1])).reshape(
                self.val_features.shape)
    ptrain = self.n_train / self.n_tot * 100
    #
    ptest = self.n_test / self.n_tot * 100
    #
    print()
    # print(
    #     'Dataset of {} points is split into training ({:.1f}%), validation ({:.1f}%), and testing ({:.1f}%).'.format(
    #         self.n_tot, ptrain, pval, ptest))

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
    # if val > 0.0:
    #     print('Val Features Shape:', self.val_features.shape)
    #     print('Val Target Shape:', self.val_target.shape)
    #     print("Train:Test:Val -->", np.round(self.train_features.shape[0] / self.feature_array.shape[0] * 100, 1), ':',
    #       np.round(self.test_features.shape[0] / self.feature_array.shape[0] * 100, 1), ":",
    #       np.round(self.val_features.shape[0] / self.feature_array.shape[0] * 100, 1))
    # else:
    #     print('Train:Test -->', np.round(self.train_features.shape[0] / self.feature_array.shape[0] * 100, 1), ':',
    #       np.round(self.test_features.shape[0] / self.feature_array.shape[0] * 100, 1))