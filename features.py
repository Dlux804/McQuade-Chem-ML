def feature_select(csv_file, model_name, selected_feat=None):
    ''' Caclulate molecular features.  Returns DataFrame and list of selected features.

    Keyword arguments:
    selected_feat -- Features you want.  Default = None (require user input)
    '''

    df, smiles_col = csvhandling.findsmiles(csv_file)
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']
    log.at[exp, 'Model Name'] = model_name
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

    return df, smiles_col, selected_feat