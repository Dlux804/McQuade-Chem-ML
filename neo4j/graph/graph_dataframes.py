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
    def model_dataframe(csv):
        """
        Objective: Create dataframe with all values and labels for machine learning results.
        :param csv: csv file
        :return: final_df: Final dataframe with all values and labels for machine learning results.
        """
        param = params.Params()  # Initiate class instance
        df, _ = param.clean_param(csv)  # Get clean param
        label_df = labels.label_model_todf(csv)  # df with all label of ml models
        final_df = pd.concat([df, label_df], axis=1)  # concat both models
        # final_df = df_drop.assign(regressor=param_clean)
        print("Final Model Dataframe for Graphing")
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
        param_df = params.Params().param_df(csv, algor)
        label_df = labels.label_param_todf(csv, algor)
        final_df = pd.concat([param_df, label_df], axis=1)
        print("Final Label Dataframe for Graphing")
        # full_labeldf.to_csv('label_test.csv')
        print(final_df)
        return final_df


# GraphDataframe().model_dataframe('ml_results3.csv')
