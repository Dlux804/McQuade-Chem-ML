"""
Written by Adam Luxon for the MPrint Project funded by the NSF.
"""

import pandas as pd
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator



datasets = ['water-energy.csv', 'log_p.csv', 'log_s.csv']


# get the data, locate smiles and exp and drop everything else
def getData(name):
    df = pd.read_csv(name)
    if name == 'log_p.csv':
        df = df.drop(['CMPD_CHEMBLID', 'ID'], axis=1)
    elif name == 'log_s.csv':
        df = df.drop(
            ['Compound ID', 'ESOL predicted log solubility in mols per litre', 'Minimum Degree', 'Molecular Weight',
             'Number of H-Bond Donors', 'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area'], axis=1)
        df = df.rename(columns={"water-sol": "exp"})
    else:
        df = df.drop(['iupac', 'calc'], axis=1)
        df = df.rename(columns={'expt': 'exp'})
    return df


# Time to featurize!
def feature_select(df, selected_feat=None):
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    if selected_feat == None:  # ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
        selected_feat = [int(x) for x in input(
            'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

    selected_feat = [feat_sets[i] for i in selected_feat]
    print("You have selected the following featurizations: ", end="   ", flush=True)
    print(*selected_feat, sep=', ')

    # This specifies what features to calculate
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
    return df

# from Github page about adding new Mprinter
def get_model_data(file_loc):
    """
    Purpose:
        Add YOUR features to csv
    Args/Requests:
         file_loc = location of csv to update
    Return:
        dataframe with updated info
    """
    print("getting file from: " + file_loc)
    all_df = pd.read_csv(file_loc)
    ### DO your stuff
    # option to pre-select what features to calculate.  If none, will ask user to pick via command line
    # likely needs additional modifications for user selection via GUI.
    feature_select(df, selected_feat=None)

    return all_df