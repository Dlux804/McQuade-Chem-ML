import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes
import pandas as pd


class GraphDataframe:
    @staticmethod
    def model_dataframe(csv):
        """

        :param csv:
        :return:
        """
        params = ep.Params()
        df, _ = params.clean_param(csv)
        label_df = ml.label_model_todf(csv)  # df with all label of ml models
        final_df = pd.concat([df, label_df], axis=1)  # concat both models
        # final_df = df_drop.assign(regressor=param_clean)
        print("Final Model Dataframe for Graphing")
        print(final_df)
        return final_df

    @staticmethod
    def param_dataframe(csv, algor):
        param_df = ep.Params().param_df(csv, algor)
        label_df = ml.label_param_todf(csv, algor)
        final_df = pd.concat([param_df, label_df], axis=1)
        print("Final Label Dataframe for Graphing")
        # full_labeldf.to_csv('label_test.csv')
        print(final_df)
        return final_df


# GraphDataframe().model_dataframe('ml_results3.csv')
