import pandas as pd
import numpy as np
import extract_params as ep
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
    def __init__(self, file):
        self.file = file

    @staticmethod
    def model_label_tolist(csv):
        """
        Objective: Create lists of all the labels I want to make from our ml results inside the csv. The goal is to make
        labels for
        :param csv:
        :return:
        """
        df = pd.read_csv(csv, index_col=0)
        df = df[df.columns.dropna()]
        col_lst = df.columns.tolist()
        col = col_lst[0:10]
        new_df = df.drop(columns=col)
        header = new_df.columns.tolist()  # List of all columns in dataframe
        label_lst = []  # List of lists of all the labels
        for i in header:
            series = new_df[i]
            lst = []
            label = i + '#'
            j = 1
            for row in series:
                new_label = label + str(j)
                j = j + 1
                lst.append(new_label)
            label_lst.append(lst)
        return header, label_lst

    @staticmethod
    def param_label_tolist(csv, algo):
        """

        :param csv:
        :return:
        """
        df = ep.param_finaldf(csv, algo)
        header = df.columns.tolist()  # List of all columns in dataframe
        label_lst = []  # List of lists of all the labels
        for i in header:
            series = df[i]
            lst = []
            label = i + '#'
            j = 1
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
        # final_df = df.assign(algorithm=algo_series)
        print(final_df)
        return final_df


def final_modeldf(header, label_lst):
    """

    :param csv:
    :return:
    """
    df = Labels.label_df(header, label_lst)
    enum_col = df["algorithmRun#"]
    header_lst = []
    for i in add_col:
        l = 1
        lst = []
        label = i + " #"
        for row in enum_col:
            new_label = label + str(l)
            l = l + 1
            lst.append(new_label)
        header_lst.append(lst)
    rotated_lst = list(rotated(header_lst))
    array_rotate = np.array(rotated_lst)
    add_df = pd.DataFrame(array_rotate, columns=add_col)
    results_df = pd.concat([df, add_df], axis=1)
    print(results_df)
    return results_df


# header, label_lst = Labels.param_label_tolist('ml_results3.csv', "gdb")
# final_modeldf(header, label_lst)
# final_df('ml_results3.csv')





