from argparse import ArgumentParser
import os
import pandas as pd
from datetime import datetime

from chemprop.features import clear_cache
from chemprop.parsing import add_train_args, modify_train_args, add_predict_args, modify_predict_args
from chemprop.train import cross_validate, make_predictions
from chemprop.utils import create_logger


class barzilayPredict:
    def cleanup_input(self, train=True, predict=True):
        if train:
            # cleanup input training data frame
            df = self.df_to_train
            lower_columns = []
            for column in df.columns:
                lower_columns.append(column.lower())
            df.columns = lower_columns
            training_df = pd.DataFrame()
            training_df['smiles'] = df['smiles']
            training_df[self.target_label] = df[self.target_label]
            training_df.to_csv('logger_dir/temp_CCS_training_df.csv', index=False)
            file_to_train = '../barzilay_predictions/logger_dir/temp_CCS_training_df.csv'
        else:
            file_to_train = None

        if predict:
            # cleanup input predicting data frame
            df = self.df_to_predict
            predicting_df = pd.DataFrame()
            lower_columns = []
            for column in df.columns:
                lower_columns.append(column.lower())
            df.columns = lower_columns
            predicting_df['smiles'] = df['smiles']
            predicting_df.to_csv('logger_dir/temp_CCS_predicting_df.csv', index=False)
            file_to_predict = '../barzilay_predictions/logger_dir/temp_CCS_predicting_df.csv'
        else:
            file_to_predict = None

        return file_to_train, file_to_predict

    def __init__(self, target_label, dataset_type, df_to_train=None, df_to_predict=None, train=True, predict=True):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/barzilay_predictions')
        self.target_label = target_label.lower()
        self.df_to_train = df_to_train
        self.df_to_predict = df_to_predict
        self.dataset_type = dataset_type
        self.save_dir = '../barzilay_predictions/logger_dir'
        self.file_to_train, self.file_to_predict = self.cleanup_input(train, predict)

        # Train dataset
        if train:
            if df_to_train is None:
                raise Exception('No training DataFrame to train with, either use train=False or pass file'
                                'through df_to_train')
            self.training_df = pd.read_csv(self.file_to_train)
            start_training_time = datetime.now()
            parser = ArgumentParser()
            add_train_args(parser)
            args = parser.parse_args([])
            args.data_path = self.file_to_train
            args.dataset_type = self.dataset_type
            args.quiet = True
            args.save_dir = '../barzilay_predictions/logger_dir'
            logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
            modify_train_args(args)
            cross_validate(args, logger)
            clear_cache()
            self.training_time = datetime.now() - start_training_time

        # Predict dataset
        if predict:
            if df_to_predict is None:
                raise Exception('No training DataFrame to predict, either use predict=False or pass file'
                                'through df_to_predict')
            self.testing_df = pd.read_csv(self.file_to_predict)
            start_predicting_time = datetime.now()
            parser = ArgumentParser()
            add_predict_args(parser)
            args = parser.parse_args([])
            args.checkpoint_dir = '../barzilay_predictions/logger_dir'
            args.preds_path = 'barzilay_predictions.csv'
            args.test_path = self.file_to_predict
            self.args = args
            modify_predict_args(self.args)
            make_predictions(self.args)
            self.predicted_results_df = pd.read_csv('barzilay_predictions.csv')
            clear_cache()
            self.predicting_time = datetime.now() - start_predicting_time
            os.chdir(cwd)

        os.chdir(cwd)
