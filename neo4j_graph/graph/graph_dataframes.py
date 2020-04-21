from graph import params  # Extract parameters from ml results
from graph import labels  # Make label dataframes
import pandas as pd

"""
    Main goal: Create a final dataframe with all values and labels for both the model and the parameters.
    Thought Process: Merge original dataframe with ml results with dataframe with corresponding labels to create a 
                     final dataframe that will be used to create knowledge graphs. 
"""


class GraphDataframe:
    @staticmethod
    def model_dataframe(csv, algor):
        """
        Objective: Create dataframe with all values and labels for machine learning results.
        :param csv: csv file
        :param algor: algorithm
        :return: final_df: Final dataframe with all values and labels for machine learning results.
        """
        param = params.Params()  # Initiate class instance
        df, _ = param.clean_param(csv)  # Get clean param
        df_algor = df[df.algorithm == algor]  # Select results with a specific algorithm
        df_reset = df_algor.reset_index(drop=True)
        label_df = labels.label_model_todf(csv, algor)  # df with all label of ml models
        drop_df = label_df.drop('algorithm', axis=1)
        # final_df = df_reset.merge(label_df)
        final_df = pd.concat([df_reset, drop_df], axis=1)  # concat both models
        print("Final Model Dataframe for Graphing")
        # final_df.to_csv('final_df.csv')
        print(final_df)
        return final_df

    @staticmethod
    def param_dataframe(csv, algor):
        """
        Objective: Create dataframe with values and labels for parameters.
        :param csv: csv file
        :param algor: algorithm
        :return: final_df: dataframe with values and labels for parameters
        """
        param = params.Params()  # Initiate class instance
        df, _ = param.clean_param(csv)  # Get clean param
        df_algor = df[df.algorithm == algor]
        clean_param = df_algor['regressor'].tolist()
        param_df = params.Params().param_df(csv, algor)
        label_df = labels.label_param_todf(csv, algor)
        concat_df = pd.concat([param_df, label_df], axis=1)
        final_df = concat_df.assign(regressor=clean_param)
        print("Final Label Dataframe for Graphing")
        # final_df.to_csv('label_test.csv')
        print(final_df)
        return final_df


# GraphDataframe().model_dataframe('ml_results3.csv')