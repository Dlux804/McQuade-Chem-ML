import pandas as pd
import numpy as np
from core.misc import cd
from core import models, regressors, analysis, features
import ast
from sklearn.model_selection import train_test_split
from time import time
from neo4j_graph.graph import params, labels
import os
import csv
from sklearn.metrics import mean_squared_error, r2_score
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

try:
   os.mkdir("./rerun_2")
except OSError as e:
   print("Directory exists")


def store(saved_item, csvfile):
    """  Organize and store model inputs and outputs.  """
    csv_name = csvfile + '.csv'
    # create dictionary of attributes
    att = dict(vars(saved_item))  # makes copy so does not affect original attributes
    del att['data']  # do not want DF in dict
    del att['smiles']  # do not want series in dict
    del att['stats']  # will unpack and add on
    # del att['impgraph']
    att.update(saved_item.stats)
    # Write contents of attributes dictionary to a CSV
    with open(csv_name, 'w', newline='') as f:  # Just use 'w' mode in Python 3.x
        w = csv.DictWriter(f, att.keys())
        w.writeheader()
        w.writerow(att)
        f.close()
    print('Storing:', csvfile)
sets = {
        'Lipophilicity-ID.csv': 'exp',
        'ESOL.csv': 'water-sol',
        'water-energy.csv': 'expt',
        'logP14k.csv': 'Kow',
        'jak2_pic50.csv': 'pIC50'
    }

def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


def parameter(data, algor):
    pre_df = pd.read_csv(data)
    df = pre_df.dropna()  # Drop rows that have empty elements. DOn't want to include a single untuned run
    algor_df = df[df.algorithm == algor]
    final_df = algor_df.reset_index(drop=True)
    param = params.Params()
    # label = labels.label_param_todf(data, algor)
    # name_lst = label['Run'].tolist()
    df = param.param_df(data, algor)
    # new_rate = df['learning_rate'].astype(float)
    col = ['algorithm', 'regressor']
    drop_df = df.drop(col, axis=1)

    for i in drop_df:
        column = drop_df[i]
        try:
            new_col = column.astype(int)
            drop_df[i] = new_col
        except ValueError:
            pass
    # drop_df['learning_rate'] = new_rate
    # print(drop_df)
    dct = drop_df.to_dict('records')
    param_lst = []
    for i in range(len(dct)):
        param_dict = dct[i]
        param_lst.append(param_dict)
    print(param_lst)
    return final_df, param_lst


def targets_features(df, exp, train=0.8, random=None):
    target = np.array(df[exp])  # exp input should be target variable string
    # remove target from features
    # axis 1 is the columns.
    features = df.drop([exp], axis=1)
    list_df = df.drop([exp, 'smiles'], axis=1)
    feature_list = list(list_df.columns)
    # features = df['smiles']
    # save list of strings of features
    # feature_list = list(features.columns)

    # convert features to numpy
    # featuresarr = np.array(features)

    train_percent = train
    test_percent = 1 - train_percent
    train_features, test_features, train_target, test_target = train_test_split(features, target,
                                                                                test_size=test_percent,
                                                                                random_state=random, shuffle=True)
    train_smiles_df = train_features.loc[:, ['smiles']]
    test_smiles_df = test_features.loc[:, ['smiles']]
    train_features = np.array(train_features.drop(['smiles'], axis=1))
    # train_features = train_features.reset_index(drop=True)
    test_features = np.array(test_features.drop(['smiles'], axis=1))

    # test_features = test_features.reset_index(drop=True)
    # feature_list = list(train_features.columns)
    return train_features, test_features, train_target, test_target, feature_list, train_smiles_df, test_smiles_df

def make_rdkit2d(df, model_name, num_feat=[0]):
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                 'morganchiral3counts', 'atompaircounts']
    if model_name == 'mlp' or model_name == 'knn':
        feat_sets.remove('rdkit2d')
        print(feat_sets)
        if num_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            num_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

        selected_feat = [feat_sets[i] for i in num_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')

    # un-normalized features are OK
    else:
        feat_sets.remove('rdkit2dnormalized')
        if num_feat == None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            num_feat = [int(x) for x in input(
                'Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
        selected_feat = [feat_sets[i] for i in num_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')

    # Start timer
    # Use descriptastorus generator
    generator = MakeGenerator(selected_feat)
    columns = []

    # get the names of the features for column labels
    for name, numpy_type in generator.GetColumns():
        columns.append(name)
    smi = df['smiles']
    print('Calculating features...', end=' ', flush=True)
    data = list(map(generator.process, smi))
    return data, columns

def main():
    os.chdir(ROOT_DIR)  # Start in root directory
    data = 'ml_results3.csv'
    algorithm = ['rf']
    for algor in algorithm:
        tuner = regressors.regressor(algor)
        df, param_list = parameter(data, algor)
        name_lst = df['Run#'].tolist()
        algo_list = df['algorithm'].tolist()
        data_list = df['dataset'].tolist()
        featmeth_list = df['feat_meth'].tolist()
        for algo, data, featmeth, param, name in zip(algo_list, data_list, featmeth_list, param_list, name_lst):
            feat_lst = ast.literal_eval(featmeth)
            sets = {
                'Lipophilicity-ID.csv': 'exp',
                'ESOL.csv': 'water-sol',
                'water-energy.csv': 'expt',
                'logP14k.csv': 'Kow',
                'jak2_pic50.csv': 'pIC50'
            }
            with cd('dataFiles'):
                print('Now in:', os.getcwd())
                target = sets[data]
                model = models.MlModel(algo, data, target)
            n = 5
            r2 = np.empty(n)
            mse = np.empty(n)
            rmse = np.empty(n)
            t = np.empty(n)
            start_time = time()
            # create dataframe for multipredict
            pva_results = pd.DataFrame()
            feats = '' + str(featmeth)
            for i in range(0, n):  # run model n times
                times = 42 + np.random.randint(10, 100)
                print('Model Type:', algo)
                print('Dataset:', data)
                print()
                model.featurization(feat_lst)
                model.regressor = tuner(**param)
                print('Parameter:', model.regressor)
                train_features, test_features, train_target, test_target, feature_list, \
                                train_smiles_df, test_smiles_df = targets_features(model.data, model.target, random=times)
                train_smiles_feat, train_smiles_columns = make_rdkit2d(train_smiles_df, algorithm)
                test_smiles_feat, test_smiles_columns = make_rdkit2d(test_smiles_df, algorithm)
                train_smiles_feat_df = pd.DataFrame(np.array(train_smiles_feat), columns=train_smiles_columns)
                test_smiles_feat_df = pd.DataFrame(np.array(test_smiles_feat), columns=test_smiles_columns)
                model.regressor.fit(train_features, train_target)
                predictions = model.regressor.predict(test_features)
                done_time = time()
                fit_time = done_time - start_time
                # Target data
                true = test_target
                # Dataframe for replicate_model
                pva = pd.DataFrame([], columns=['actual', 'predicted'])
                pva['actual'] = true
                pva['predicted'] = predictions
                r2[i] = r2_score(pva['actual'], pva['predicted'])
                pva_results['r2'] = r2
                print('R2', r2)
                mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
                pva_results['mse'] = mse
                rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
                pva_results['rmse'] = rmse
                t[i] = fit_time
                # store as enumerated column for multipredict
                print('Prediction length:', len(predictions))
                # pva_multi['predicted' + str(i)] = predictions

                with cd('rerun_2'):
                    train_smiles_df.to_csv('train_smiles' + '_' + model.dataset[:-4] + '_' + model.algorithm + '_' +
                                     feats + '_' + str(i) + '.csv')
                    train_smiles_feat_df.to_csv('feat_train_smiles' + '_' + model.dataset[:-4] + '_' + model.algorithm + '_' +
                                     feats + '_' + str(i) + '.csv')
                    test_smiles_df.to_csv('test_smiles' + '_' + model.dataset[:-4] + '_' + model.algorithm + '_' +
                                           feats + '_' + str(i) + '.csv')
                    test_smiles_feat_df.to_csv('feat_test_smiles' + '_' + model.dataset[:-4] + '_' + model.algorithm + '_' +
                                     feats + '_' + str(i) + '.csv')
                    # pva_results.to_csv('results' + '_' + model.dataset[:-4] + '_' + model.algorithm + '_' +
                    #                        feats + '_' + str(i) + '.csv')

                print()
            # pva_multi['pred_avg'] = pva.mean(axis=1)
            # pva_multi['pred_std'] = pva.std(axis=1)
            # pva_multi['actual'] = test_targe

            model.stats = {
                'r2': list(r2),
                'r2_avg': r2.mean(),
                'r2_std': r2.std(),
                'mse': list(mse),
                'mse_avg': mse.mean(),
                'mse_std': mse.std(),
                'rmse': list(rmse),
                'rmse_avg': rmse.mean(),
                'rmse_std': rmse.std(),
                'time_avg': t.mean(),
                'time_std': t.std()
            }
            print('Average R^2 = %.3f' % model.stats['r2_avg'], '+- %.3f' % model.stats['r2_std'])
            print('Average RMSE = %.3f' % model.stats['rmse_avg'], '+- %.3f' % model.stats['rmse_std'])
            print()
            print()

            feats = '' + '-' + str(featmeth)

            # create model file name
            name = model.dataset[:-4] + '-' + model.algorithm + feats
            # store(model, name)
            with cd('rerun_2'):
                # smiles_df.to_csv('smiles'+name+n+'.csv')
                store(model, name)


# with cd('rerun_3'):
#     new = pd.DataFrame()
#     df1 = pd.read_csv('test_smiles_ESOL_rf_[0, 2]_0.csv')
#     df2 = pd.read_csv('test_smiles_ESOL_rf_[0, 2]_1.csv')
#     df1['MatchSmiles'] = np.where(df1['smiles'] == df2['smiles'], 'True', 'False')
#     df1.to_csv('match_smiles.csv')
#     print(df1.equals(df2))
    # dct = df.to_dict('records')
    # for i in range(len(dct)):  # Enumerate over the dictionary
    #     row_dict = dct[i]  # Enumerate over every row
    #     params = row_dict['r2']  # Enumerate over every cell in the regressor column
    #     reg = params[params.find("[") + 1:params.find("]")]
    #     print(reg.split())
    #     final_reg = reg.replace(" ", ", ")

if __name__ == "__main__":
    main()



