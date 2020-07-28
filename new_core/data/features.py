"""
Objective: Featurize the data
"""

from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import pandas as pd
from time import time, sleep
from tqdm import tqdm
from rdkit import Chem
from new_core.data.ingest import Ingest


class Feature:
    def __init__(self, dataset, target, feat_meth):
        """

        :param data:
        :param feat_meth:
        """
        self.dataset = dataset
        self.target = target
        self.feat_meth = feat_meth

    def featurize(self, not_silent=True):
        """
        Caclulate molecular features.
        Returns DataFrame, list of selected features (numeric values. i.e [0,4]),
         and time to featurize.
        Keyword arguments:
        feat_meth -- Features you want by their numerical value.  Default = None (require user input)
        """
        self.data, smiles_col = Ingest(self.dataset, self.target).load_smiles()
        feat_meth = self.feat_meth
        df = self.data

        # available featurization options
        feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts',
                     'morganchiral3counts', 'atompaircounts']

        if feat_meth is None:  # ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)]
            feat_meth = [int(x) for x in input(
                """Choose your features  by number from list above.  
                                                        You can choose multiple with \'space\' delimiter:  """).split()]
        selected_feat = [feat_sets[i] for i in feat_meth]
        self.feat_method_name = selected_feat

        # Get data from MySql if called
        # if retrieve_from_mysql:
        #     print("Pulling data from MySql")
        #     self.featurize_from_mysql()
        #     return
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
            if x is None:
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
        df = pd.concat([df, features], axis=1)
        df = df.dropna()

        # remove the "RDKit2d_calculated = True" column(s)
        df = df.drop(list(df.filter(regex='_calculated')), axis=1)
        df = df.drop(list(df.filter(regex='[lL]og[pP]')), axis=1)

        # store data back into the instance
        self.data = df
        self.feat_time = feat_time
        # return df, feat_time,
