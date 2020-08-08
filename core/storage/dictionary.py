"""
Objective: This file contains different dictionaries that are used throughout the pipeline
"""


def target_name_grid(dataset):
    """
    Objective: Return true experimental target for a dataset
    :param dataset:
    :return:
    """
    grid = {
            'ESOL.csv': 'water_sol',
            'Lipophilicity-ID.csv': 'logP',
            'water-energy.csv': 'hydration_energy',
            'logP14k.csv': 'logP_kow',
            'jak2_pic50.csv': 'pIC50',
            'Lipo-short.csv': 'logP'
        }
    return grid[dataset]
