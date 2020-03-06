import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes
import pandas as pd


class GraphDataframe:
    @staticmethod
    def model_dataframe(csv, algor):
        """

        :param csv:
        :param algor:
        :return:
        """
        _, _, param_clean = ep.Params().param_extract(csv, algor)
        df = pd.read_csv(csv)  # df of original csv files with all ml results
        model_df = df[df.tuneTime != 0]
        label_modeldf = ml.label_model_todf(csv)  # df with all label of ml models
        full_modeldf = pd.concat([model_df, label_modeldf], axis=1)  # concat both models
        algo_modeldf = full_modeldf[full_modeldf.algorithm == algor]  # dataframe with specific algorithms
        df_drop = algo_modeldf.drop(columns="regressor")
        final_df = df_drop.assign(regressor=param_clean)
        print("Final Model Dataframe for Graphing")
        print(final_df)
        return final_df

    @staticmethod
    def label_dataframe(csv, algor):
        param_df = ep.Params().param_df(csv, algor)
        label_paramdf = ml.label_param_todf(csv, algor)
        full_labeldf = pd.concat([param_df, label_paramdf], axis=1)
        print("Final Label Dataframe for Graphing")
        print(full_labeldf)
        return full_labeldf
