import pandas as pd
import numpy as np
from graph import params
"""
    Main goal: Make dataframes that will contain labels for the ml results and paramters in the csv containing machine
                learning results. We will then use these labels to create nodes for our knowledge graphs.
    Thought Process: We use the column headers from our csv as the basis to make our labels. First, for every n runs, 
                     we need an "#n" attached to the header as values for our new dataframe. Then we attach "Run#" to 
                     the headers to make new headers for said dataframe. We need to make a dataframe of that kind for
                     the whole model and for the tuned parameters
                     Example: r2_avgRun# (Column headers): r2_avg#0, r2_avg#1, r2_avg#n, ... (Column values) 
                    
                     . 
"""


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
        Objective: Create lists of all the labels and headers from our ml results' csv.
        :param csv: csv file
        :return header: List of all columns we need to make the dataframe that contains all the labels
        :return label_lst: List of lists of all the labels created from the headers.
                           This will later be used to make a dataframe

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
            lst = []  # list for every enumeration
            label = i + '#'  # Attach a # symbol to the header name
            j = 0
            for row in series:  # Enumerate over all values of a column
                new_label = label + str(j)
                j = j + 1
                lst.append(new_label)  # append list for every enumeration
            label_lst.append(lst)  # List of lists for every enumeration
        return header, label_lst

    @staticmethod
    def param_label_tolist(csv, algor):
        """

        Objective: Create lists of all the labels and headers specifically for parameters from our ml results' csv.
        :param csv: csv file
        :param algor: name of the algorithm
        :return header: List of all columns we need to make the dataframe that contains all the labels
        :return label_lst: List of lists of all the labels created using the header.
                           This will later be used to make a dataframe
        """
        param = params.Params()  # Initiate class intance
        df = param.param_df(csv, algor)  # Acquire dataframe with all the necessary column headers
        header = df.columns.tolist()  # List of all columns in dataframe  # Get the header
        # print(header)
        label_lst = []  # List of lists of all the labels
        for i in header:  # Enumerate over all column headers
            series = df[i]
            lst = []  # list for every enumeration
            label = i + '#'  # Attach a # symbol to the header name
            j = 0
            for row in series:  # Enumerate over all values of a column
                new_label = label + str(j)
                j = j + 1
                lst.append(new_label)  # append list for every enumeration
            label_lst.append(lst)  # List of lists for every enumeration
        return header, label_lst

    @staticmethod
    def label_df(header, label_lst):
        """
        Objective: Put headers and labels together into one dataframe
        :param header: list of all the column headers
        :param label_lst: list of all the labels created from the headers
        :return:
        """
        rotated_lst = list(rotated(label_lst))  # Rotate list of lists -90 degrees so it can line up with the columns
        array_rotate = np.array(rotated_lst)  # Turn into array
        header_lst = []  # List of all the header
        for i in header:
            new_item = i + 'Run#'  # Attach a Run# to all the header
            header_lst.append(new_item)  # Append new header list
        final_df = pd.DataFrame(array_rotate, columns=header_lst)  # Make dataframe using the created labels and headers
        return final_df


def label_model_todf(csv):
    """
    Objective: Add the Run# and Result# column to the dataframes with all the labels for ml results
    :param csv: csv file
    :return: results_df: This is the final dataframe that contains all the labels needed to make the knowledge graph
                            for ml results
    """
    add_col = ["Run", "Results"]  # Names of the two columns we wish to add
    header, label_lst = Labels.model_label_tolist(csv)  # Create headers and labels lists
    df = Labels.label_df(header, label_lst)  # Make a dataframe that contains all the labels created
    enum_col = df["algorithmRun#"]  # Use one of the columns in the dataframe to enumerate
    header_lst = []  # New header list
    for i in add_col:  # Enumerate over the 2 new headers
        num = 0  # Start at 0
        lst = []  # List that will contain new labels
        label = i + " #"  # Add the symbol #
        for row in enum_col:  # Enumerate over the number of rows that we have
            new_label = label + str(num)  # Add the number
            num = num + 1  # Add 1
            lst.append(new_label)  # Append lables into list
        header_lst.append(lst)  # Make list of lists
    rotated_lst = list(rotated(header_lst))  # Rotate list of lists
    array_rotate = np.array(rotated_lst)
    add_df = pd.DataFrame(array_rotate, columns=add_col)  # Make a dataframe using the new labels
    results_df = pd.concat([df, add_df], axis=1)  # Concat 2 dataframes to make a master dataframe
    # final_df = results_df.assign(algorithm=algo_lst)
    # print(results_df)
    return results_df


def label_param_todf(csv, algor):
    """

    Objective: Add the Run# and Result# column to the dataframes with all the labels for parameters
    :param csv: csv file
    :param algor: algorith
    :return: results_df: This is the final dataframe that contains all the labels needed to make the knowledge graph
                            for parameters
    """
    add_col = ["Run", "Results"]  # Names of the two columns we wish to add
    header, label_lst = Labels.param_label_tolist(csv, algor)  # Create headers and labels lists
    df = Labels.label_df(header, label_lst)  # Make a dataframe that contains all the labels created
    enum_col = df["algorithmRun#"]  # Use one of the columns in the dataframe to enumerate
    header_lst = []  # New header list
    for i in add_col:  # Enumerate over the 2 new headers
        num = 0  # Start at 0
        lst = []  # List that will contain new labels
        label = i + " #"  # Add the symbol #
        for row in enum_col:  # Enumerate over the number of rows that we have
            new_label = label + str(num)  # Add the number
            num = num + 1  # Add 1
            lst.append(new_label)  # Append lables into list
        header_lst.append(lst)  # Make list of lists
    rotated_lst = list(rotated(header_lst))  # Rotate list of lists
    array_rotate = np.array(rotated_lst)
    add_df = pd.DataFrame(array_rotate, columns=add_col) # Make a dataframe using the new labels
    results_df = pd.concat([df, add_df], axis=1)  # Concat 2 dataframes to make a master dataframe
    # print(results_df)
    # results_df.to_csv('test_gdb.csv')
    return results_df
