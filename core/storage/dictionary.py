"""
Objective: This file contains different dictionaries that are used throughout the pipeline
"""


def target_name_grid(dataset):
    """"""
    grid = {
            'ESOL.csv': 'water_sol',
            'Lipophilicity-ID.csv': 'logP',
            'water-energy.csv': 'hydration_energy',
            'logP14k.csv': 'logP',
            'jak2_pic50.csv': 'pIC50',
            'Lipo-short.csv': 'logP'
        }
    return grid[dataset]
