import os
from core import misc
from datetime import datetime

"""
Objective: A systematic way to create a unique name for every single machine learning run
This code follows the "Letters and Numbers" rule explained in naming_schemes_v2.pptx in the naming branch
"""


def represent_dataset():
    """
    Get string representation for all current dataset in dataFiles
    This representation follows the "Letters and Numbers" rule explained in naming_schemes_v2.pptx in the naming branch
    :return: Dictionary of dataset as key and their representation as value
    """
    with misc.cd('C:/Users/quang/McQuade-Chem-ML/dataFiles'):  # Access folder with all dataset
        for roots, dirs, files in os.walk(os.getcwd()):
            data_dict = {}  # Dictionary of dataset and their first character
            for dataset in files:  # Loop through list of files
                if not dataset[0].isdigit():  # If first letter is not a number
                    data_dict[dataset] = dataset[0]  # Key as dataset as first character as value
                else:  # If first letter is a number
                    newstring = ''  # Empty string
                    for letter in dataset:  # Start looping through every character in the name
                        if letter.isdigit():  # If the character is a digit
                            newstring += letter  # Add number to the empty string
                        else:  # If letter is a string
                            newstring += letter  # Add string to the empty string
                            data_dict[dataset] = newstring
                            break  # Stop sequence at the first letter
    compare = []  # This list is for dataset that have unique first character
    repeat = []  # This list is for dataset that have matching first character
    duplicate_dict = {}  # Dictionary of dataset with the same first character
    for key in data_dict:  # For dataset in dictionary
        for value in data_dict[key]:  # For first character
            if value not in compare:  # If first character is not in our list
                compare.append(value)  # Add it to the empty list
            else:
                repeat.append(key)  # Add dataset that has matching first character to list
                duplicate_dict[value] = repeat  # Key as the string and list of dataset as values
    unique_list = []
    for key in duplicate_dict:  # For every key in duplicate dictionary
        count = 1  # Counter starts at 1
        unique = {}  # Final dictionary with all unique string representation for their respective dataset
        for duplicate in duplicate_dict[key]:  # For dataset with matching dataset_string
            data_dict.pop(duplicate, None)  # Remove values that have unique first character
            dataset_string = ''.join([duplicate[:-4][0], duplicate[:-4][-count:]])  # Combing first and last character
            if dataset_string not in unique.values():  # Check to see if the newly created string has duplicate
                unique[duplicate] = dataset_string   # Key as the dataset and the newly created string as value
            else:  # If the string still has duplicate
                count += 1  # Increase counter by 1
                dataset_string = ''.join([duplicate[:-4][0], duplicate[:-4][-count:]])  # First, last and second to last
                unique[duplicate] = dataset_string  # Key as the dataset and the newly created string as value
                break  # Break the loop
        unique_list.append(unique)  # Get all dictionary for a situation that has multiple matching first character
    for dictionary in unique_list:  # Loop through all dictionaries
        data_dict.update(dictionary)  # Update the original dictionary
    return data_dict


def represent_algorithm():
    """
    Get string representation for all currently supported regressors
    This representation follows the "Letters and Numbers" rule explained in naming_schemes_v2.pptx in the naming branch
    :return:
    """
    algorithm_list = ['ada', 'rf', 'svr', 'gdb', 'mlp', 'knn']  # Out current supported algorithm
    represent = [algor[0].upper() for algor in algorithm_list]  # List of algorithm's first letter with upper case
    dictionary = {}
    for algor, rep in zip(algorithm_list, represent):  # Looping through two lists at the same time
        dictionary[algor] = rep  # Key as algorithm and their string representation as value
    return dictionary


def name(algorithm, dataset, feat_meth, tune=False):
    """
    Give a unique name to a machine learning run
    :param algorithm: Name of algorithm
    :param dataset: Name of dataset
    :param feat_meth: List of feature method
    :param tune: Tune or not
    :return: A unique name to a machine learning run
    """
    algorithm_dict = represent_algorithm()  # Get dictionary of algorithm and their string representation
    dataset_dict = represent_dataset()  # Get dictionary of dataset and their string representation
    if algorithm in algorithm_dict.keys():  # If the input algorithm is supported
        algorithm_string = algorithm_dict[algorithm]  # Get the string representation
    else:  # If it is not supported, give an error
        raise ValueError("The algorithm of your choice is not supported in our current workflow. Please use the "
                         "algorithms offered in grid.py ")
    if dataset in dataset_dict.keys():  # If input dataset is one that we have
        dataset_string = dataset_dict[dataset]  # Get the string representation
    else:  # If new dataset
        count = 2
        if dataset[:-4][0] in dataset_dict.values():  # If this dataset has the same first character as existing ones
            print("Duplicate First letter in input. Adding last and second to last characters")
            # Add first, last and second to last just to be safe
            dataset_string = ''.join([dataset[:-4][0], dataset[:-4][-count:]])
        else:  # If this dataset has a unique first character (compared to what we have in dataFiles)
            dataset_string = dataset[:-4][0]  # Dataset string will be its first character
    feat_string = ''.join(map(str, feat_meth))
    if tune:
        tune_string = str(1)  # If the model is tuned, tune string will be 1
    else:
        tune_string = str(0)  # Else, it will be 0
    now = datetime.now()
    date_string = now.strftime("%y%m%d%H%M%S_")  # Get date and time string
    run_name = ''.join([date_string, algorithm_string, dataset_string, feat_string, tune_string])  # Run name
    date = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Created {0} on {1}".format(run_name, date))
    return run_name
