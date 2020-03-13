import pandas as pd
import re
import numpy as np

"""
    Main Goal:  Extract tuned parameters so they can be used as additional resources for our knowledge graph.
    Thought process: Firstly, I Extract parameters from the "regressor" column of the ml results csv. Then take out 
                      the parameters that are unique and put them into a dataframe that can later be used to add into
                      our knowledge graph.
                  
"""

col_gdb = ['alpha', 'ccp_alpha', 'criterion', 'init', 'learning_rate', 'loss', 'max_depth', 'max_features',
           'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split',
           'min_weight_fraction_leaf', 'n_estimators', 'n_iter_no_change', 'presort', 'random_state', 'subsample',
           'tol', 'validation_fraction', 'verbose', 'warm_start']  # All of GDB's parameters

col_rf = ['bootstrap', 'ccp_alpha', "criterion", 'max_depth', "max_features",
          'max_leaf_nodes', 'max_samples', 'min_impurity_decrease', 'min_impurity_split',
          'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators',
          'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start']  # All of RF's parameters

# All of KNN's parameter
col_knn = ["algorithm", 'leaf_size', "metric", 'metric_params', 'n_jobs', 'n_neighbors', 'p', "weights"]

col_ada = ['base_estimator', 'learning_rate', "loss", 'n_estimators', 'random_state']  # All of Ada's parameter

# Dictionary that match the algorithm with the correct parameter
def param_dict(algor):
    params_dict = {
        "gdb": col_gdb,
        "rf": col_rf,
        "knn": col_knn,
        "ada": col_ada
    }
    return params_dict[algor]


def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


class Params:
    @staticmethod
    def clean_param(csv):
        """
        Extract parameters and put them into a list. Also put in a new regressor column that can help us merge knowledge
        graphs together. While the parameters are separated, they still have "=" signs on them.
        :param csv: a csv file
        :return: finaL_df: clean dataframe with new regressor column
        :return: param_list: List of all separated parameters
        """
        pre_df = pd.read_csv(csv, index_col=0)  # Read in csv, first column as index so there wont be any unnamed column
        df_clean = pre_df.dropna()  # Drop rows that have empty elements. DOn't want to include a single untuned run
        df = df_clean.reset_index(drop=True)  # Reset index to fil the gap
        # df_algor = df[df.algorithm == algor]
        dct = df.to_dict('records')  # Store dataframe as as a record type dictionary
        param_list = []  # List of all parameters extracted
        param_clean = []  # List of parameters that don't have the '' sign
        for i in range(len(dct)):  # Enumerate over the dictionary
            row_dict = dct[i]  # Enumerate over every row
            params = row_dict['regressor']  # Enumerate over every cell in the regressor column
            # print(params)
            reg = params[params.find("(") + 1:params.find(")")]  # Find the parameters located between parentheses
            new_reg = " ".join(reg.split())  # Remove large spacing between parameters
            final_reg = new_reg.replace("'", "")  # Remove the '' off some of the parameters
            element = final_reg.split(",")  # Take out the parameters separated by commas
            param_list.append(element)  # Store all the parameters into list
            param_clean.append(final_reg)  # Store the parameters that no longer have ''
        drop_df = df.drop(columns='regressor')  # Drop the original regressor
        final_df = drop_df.assign(regressor=param_clean)  # Put in a regressor with clean parameters into the dataframe
        return final_df, param_list

    @staticmethod
    def param_lst(csv, algor):
        """
        Objective: In this script, we remove the parameters that are not unique, then we remove the "=" signs out of the
                    parameter, leaving only the name.
        :param csv: csv file
        :param algor: algorithm name
        :return: col_list: list of all the header
        :return: main_list: list of list of all the cleaned parameters
        """
        pre_df, param_list = Params.clean_param(csv)  # Get dataframe and parameter list from clean_param
        df = pre_df.assign(param_lst=param_list)  # Assign the the param list to the dataframe
        df_algor = df[df.algorithm == algor]  # Choose the rows with the corresponding algorithm
        param_list = df_algor["param_lst"].tolist()
        algor_lst = df_algor["algorithm"].tolist()
        regress_lst = df_algor['regressor'].tolist()
        param_df = pd.DataFrame.from_records(param_list, columns=param_dict(algor))  # Attach param_list to a dataframe
        unique_df = param_df.loc[:, ~(param_df == param_df.iloc[0]).all()]  # Only keep unique parameters in dataframe
        # Assign algorithm and regressor columns to dataframe
        final_df = unique_df.assign(algorithm=algor_lst, regressor=regress_lst)
        # print(df_with_algor)
        col_list = final_df.columns.tolist()  # Put headers of new data frame into list
        main_list = []  # Parameter's list of list
        for col in col_list:  # Enumerate over column headers
            col_lst = []  # Temporary list
            for i in final_df[col]:  # Enumerate over the column headers in the datafrmae
                t = re.sub('.*=', '', i)  # Remove the = sign
                col_lst.append(t)
            main_list.append(col_lst)
        return col_list, main_list

    def param_df(self, csv, algor):
        """
        Objective: Return a dataframe with all extracted, cleaned and uniquely tuned parameters
        :param csv: csv file
        :param algor: algorithm name
        :return: df_records: Final dataframe with all parameters
        """
        col_list, main_list = Params.param_lst(csv, algor)  # get column headers and values
        rotate_lst = list(rotated(main_list))  # Rotate the list of lists
        array_rotate = np.array(rotate_lst)
        df_records = pd.DataFrame.from_records(array_rotate, columns=col_list)  # Put everything into a dataframe
        # final_df.to_csv("test.csv")
        return df_records

# params = Params()
# params.param_extract('ml_results3.csv')
# params.param_lst('ml_results3.csv', "gdb")
# params.param_df('ml_results3.csv', "gdb")

