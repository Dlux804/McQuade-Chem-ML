import pandas as pd
import numpy as np
from core.misc import cd
from rerun_2 import merge_csv
import os
import csv

name_list = merge_csv.csv_names()


def concat_smiles_df(starting_string):
    for name in name_list:
        print(name)
        all_df = []
        try:
            with cd(name):
                for root, dirs, files in os.walk(r'C:/Users/quang/McQuade-Chem-ML/' + name):
                    for f in files:
                        if f.startswith(starting_string):
                            df = pd.read_csv(f)
                            df = df.drop('Unnamed: 0', 1)
                            df['file'] = f.split('/')[-1]
                            all_df.append(df)
                    merge_all_df = pd.concat(all_df, ignore_index=False, axis=1)
                    merge_all_df.to_csv('merged' + '_' + starting_string + '_' + name + '_' + '.csv', index=False)
                    merge_all_df = pd.read_csv('merged' + '_' + starting_string + '_' + name + '_' + '.csv',
                                               mangle_dupe_cols=True)
                    merge_all_df.to_csv('merged' + '_' + starting_string + '_' + name + '_' + '.csv', index=False)
        except FileNotFoundError:
            pass

# concat_smiles_df('train')
def unique(lst):
    res = []
    for i in lst:
        if i not in res:
            res.append(i)
        else:
            pass
    return res

def compare_feat_value(starting_string):
    for name in name_list:
        # print(name)
        all_df = []
        data = dict()
        try:
            with cd(name):
                for root, dirs, files in os.walk(r'C:/Users/quang/McQuade-Chem-ML/' + name):
                    print('In dir:', name)
                    count = 0
                    col_list = []
                    mean_list = []
                    for f in files:
                        if f.startswith(starting_string):
                            df = pd.read_csv(f)
                            clean_df = df.drop(['RDKit2D_calculated', 'Unnamed: 0'], axis=1)
                            df_mean = clean_df.mean(axis=0)
                            to_df = df_mean.to_frame()
                            final_df = to_df.transpose()
                            print('In file:', f)
                            print(final_df)
                    final_df.to_csv('mean' + '_' + starting_string + '_' + name + '.csv')
                    #         mean_list.append(df_mean)
                    # print(mean_list)
                            # for col in clean_df.columns:
                            #     col_list.append(col)
                            # unique_col = unique(col_list)
                            # mean_list = []
                            # for feat in unique_col:
                            #     mean_col = clean_df[feat].mean()
                            #     print('Calculating mean for:', feat)
                            #     print('In file:', f)
                            #     print(mean_col)
                            #     mean_list.append(mean_col)
                            # mean_dict = dict(zip(unique_col, mean_list))
                            # new_df = pd.DataFrame(mean_dict, index=[0])
                            # new_df.to_csv('test'+'_'+name+'.csv')


        except FileNotFoundError:
            pass

# feat = ['feat_test', 'feat_train']
# for i in feat:
#     compare_feat_value(i)

