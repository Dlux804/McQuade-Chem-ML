from time import time, sleep

import pandas as pd
import numpy as np
from tqdm import tqdm
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from rdkit import Chem


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
    if self.dataset == "flashpoint.csv":
        self.data['flashpoint'] = [float(i) for i in list(self.data['flashpoint'])]
    df = self.data
    # df = canonical_smiles(df)  # Turn SMILES into CANONICAL SMILES

    # print(df[df.isna().any(axis=1)])
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

    df = df[~df.index.duplicated(keep='first')]
    features = features[~features.index.duplicated(keep='first')]
    df = pd.concat([df, features], axis=1)
    df = df.dropna()

    # remove the "RDKit2d_calculated = True" column(s)
    df = df.drop(list(df.filter(regex='_calculated')), axis=1)
    df = df.drop(list(df.filter(regex='[lL]og[pP]')), axis=1)

    # store data back into the instance
    self.data = df
    self.feat_time = feat_time
    self.data.iloc[:, 1:] = self.data.iloc[:, 1:].apply(pd.to_numeric)
    # Replacing infinite with nan
    self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Dropping all the rows with nan values
    self.data.dropna(inplace=True)
    self.data.reset_index(drop=True, inplace=True)
