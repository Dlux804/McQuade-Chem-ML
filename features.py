from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd

def feature_select(df, model_name, selected_feat=None):
    """ Caclulate molecular features.  Returns DataFrame and list of selected features.

    Keyword arguments:
    selected_feat -- Features you want.  Default = None (require user input)
    """

    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']

    # Remove un-normalized feature option depending on model type
    if model_name == 'nn' or model_name == 'knn':
        feat_sets.remove('rdkit2d')
        print(feat_sets)
        if selected_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
            selected_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

        selected_feat = [feat_sets[i] for i in selected_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')
    else:
        if selected_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
            selected_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
        selected_feat = [feat_sets[i] for i in selected_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')

    # Use descriptastorus generator
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

    return df, selected_feat