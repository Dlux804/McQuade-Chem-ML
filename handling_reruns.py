import pandas as pd
import numpy as np
from core.misc import cd
from rerun_2 import merge_csv
import os
import csv
import itertools

name_list = merge_csv.csv_names()

lst = ['0', '1', '2', '3', '4']
combo_lst = []
for a, b in itertools.combinations(lst, 2):
    combo_lst.append(a+b)
# print(new_lst)

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
                    # final_df.to_csv('mean' + '_' + starting_string + '_' + name + '.csv')
        except FileNotFoundError:
            pass
# compare_feat_value('feat_test')


def similar_smiles(starting_string):
    for name in name_list:
        # print(name)
        try:
            with cd(name):
                for root, dirs, files in os.walk(r'C:/Users/quang/McQuade-Chem-ML/' + name):
                    print('In dir:', name)
                    df_list = []
                    for f in files:
                        if f.startswith(starting_string):
                            print('Working with:', f)
                            df = pd.read_csv(f)
                            df_list.append(df)
                        similarity_lst = []
                        for df1, df2 in itertools.combinations(df_list, 2):
                            similarity_df = pd.merge(df1, df2, on='smiles')
                            similarity_df = similarity_df.drop([ 'Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
                            print(similarity_df)
                            similarity_lst.append(similarity_df)
                        # combination_lst = ['01', '02', '03']
                    for similar, combo in zip(similarity_lst, combo_lst):
                        similar.to_csv('overlap_smiles' + '_' + starting_string + '_' + name + '_' +
                                           combo + '.csv')
                            # test_df.to_csv('test' + '_' + name + '.csv')
        except FileNotFoundError:
            pass

similar_smiles('final_feat_train_smiles')

def concat_csvs(starting_string):
    for name in name_list:
        # print(name)
        try:
            with cd(name):
                for root, dirs, files in os.walk(r'C:/Users/quang/McQuade-Chem-ML/' + name):
                    print('In dir:', name)
                    starting_string_list = []
                    feat_list = []
                    for f in files:
                        if f.startswith(starting_string):
                            print('Working with starting:', f)
                            starting_string_list.append(f)
                        elif f.startswith('feat_' + starting_string):
                            print('Working with feat:', f)
                            feat_list.append(f)
                    final_df_list = []
                    for starting_csv, feat_csv in zip(starting_string_list, feat_list):
                        df1 = pd.read_csv(starting_csv)
                        df2 = pd.read_csv(feat_csv)
                        final_df = pd.concat([df1, df2], axis=1)
                        final_df = final_df.drop('Unnamed: 0', axis=1)
                        final_df_list.append(final_df)
                    for file, final_df in zip(starting_string_list, final_df_list):
                        final_df.to_csv('final_' + 'feat_' + file + '.csv')

        except FileNotFoundError:
            pass

# concat_csvs('test_smiles')