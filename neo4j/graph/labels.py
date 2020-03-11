import pandas as pd
import numpy as np
from graph import params
"""
Objective: Make dataframes that will contain names 
            and labels for knowledge graphs using a dataframe that only has values 
"""

add_col = ["Run", "Results"]


def rotated(array_2d):
    """
    Flip list 90 degrees to the left
    :param array_2d:
    :return: a list that is turned 90 degrees to the left
    """
    list_of_tuples = zip(*reversed(array_2d[::-1]))
    return [list(elem) for elem in list_of_tuples]


class Labels:

    @staticmethod
    def model_label_tolist(csv):
        """
        Objective: Create lists of all the labels I want to make from our ml results inside the csv. The goal is to make
        labels for the csv that contains ml(machine learning) results.
        :param csv: csv file
        :return:
        """
        pre_df = pd.read_csv(csv)  # Read csv
        df = pre_df.dropna()  # Drop empty rows("Untuned results).
        # df.to_csv('drop.csv')
        algo_lst = df["algorithm"].tolist()  # I want to keep the algorithm column so we can select rows by algorithm
        col_lst = df.columns.tolist()  # Turn df headers to list
        col = col_lst[0:10]  # Unused column
        new_df = df.drop(columns=col)  # Drop unused column
        new_df["algorithm"] = algo_lst  # Attach the column that contains all algorithms
        # print(new_df)
        header = new_df.columns.tolist()  # List of all columns that we need in the dataframe
        # print(header)
        label_lst = []  # List of lists of all the labels
        for i in header:  # Enumerate over all column headers
            series = new_df[i]
            lst = []
            label = i + '#'  # Attach a # symbol to the header name
            j = 0
            for row in series:  # Enumerate over all values of a column
                new_label = label + str(j)
                j = j + 1
                lst.append(new_label)  # list for every enumeration
            label_lst.append(lst)  # Master list
        return header, label_lst

    @staticmethod
    def param_label_tolist(csv, algor):
        """

        :param csv:
        :param algor:
        :return:
        """
        params = params.Params()
        df = params.param_df(csv, algor)
        header = df.columns.tolist()  # List of all columns in dataframe
        # print(header)
        label_lst = []  # List of lists of all the labels
        for i in header:
            series = df[i]
            lst = []
            label = i + '#'
            j = 0
            for row in series:
                new_label = label + str(j)
                j = j + 1
                lst.append(new_label)
            label_lst.append(lst)
        return header, label_lst

    @staticmethod
    def label_df(header, label_lst):
        """

        :param header:
        :param label_lst:
        :return:
        """
        rotated_lst = list(rotated(label_lst))
        array_rotate = np.array(rotated_lst)
        header_lst = []
        for i in header:
            new_item = i + 'Run#'
            header_lst.append(new_item)
        final_df = pd.DataFrame(array_rotate, columns=header_lst)
        return final_df


def label_model_todf(csv):
    """

    :param csv:
    :return:
    """
    header, label_lst = Labels.model_label_tolist(csv)
    df = Labels.label_df(header, label_lst)
    enum_col = df["algorithmRun#"]
    header_lst = []
    for i in add_col:
        num = 0
        lst = []
        label = i + " #"
        for row in enum_col:
            new_label = label + str(num)
            num = num + 1
            lst.append(new_label)
        header_lst.append(lst)
    rotated_lst = list(rotated(header_lst))
    array_rotate = np.array(rotated_lst)
    add_df = pd.DataFrame(array_rotate, columns=add_col)
    results_df = pd.concat([df, add_df], axis=1)
    # final_df = results_df.assign(algorithm=algo_lst)
    # print(results_df)
    return results_df


def label_param_todf(csv, algor):
    """

    :param csv:
    :param algor
    :return:
    """
    header, label_lst = Labels.param_label_tolist(csv, algor)
    df = Labels.label_df(header, label_lst)
    enum_col = df["algorithmRun#"]
    header_lst = []
    for i in add_col:
        num = 0
        lst = []
        label = i + " #"
        for row in enum_col:
            new_label = label + str(num)
            num = num + 1
            lst.append(new_label)
        header_lst.append(lst)
    rotated_lst = list(rotated(header_lst))
    array_rotate = np.array(rotated_lst)
    add_df = pd.DataFrame(array_rotate, columns=add_col)
    results_df = pd.concat([df, add_df], axis=1)
    # print(results_df)
    # results_df.to_csv('test_gdb.csv')
    return results_df


# header, label_lst = Labels.model_label_tolist('ml_results3.csv')
# label_param_todf('ml_results3.csv', "gdb")

# label_model_todf('ml_results3.csv')
