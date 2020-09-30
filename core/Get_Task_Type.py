""" This function allows for main.py to skip over algorithm/data set combinations that are not compatible.

The checker variable is set to 0 if the combination is not compatible, and 1 otherwise.

It is then returned to main.py, where it is used to decide if the model/data set combination should be ran,
or the combination should be skipped.

This function also allows for the task type to be returned, so that it can be printed in main.py.
"""
def Get_Task_Type_1(data, alg):
    if data in ['sider.csv', 'clintox.csv'] and alg in ['svm', 'ada', 'gdb']:  # SVC, ada, and gdb are not compatible with multi-label classification.
        checker = 0

        # These classification data sets are not compatible with these regression models
    elif data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv'] and alg in ['nn']:
        checker = 0

        # These regression data sets are not compatible with these classification models.
    elif data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv'] and alg in []:
        checker = 0

    else: # Checker = 1 when there are no compatibility issues
        checker = 1

    # The following if statements determine the task type based on the data set being used.
    # task_type is used in main.py to print what type of task is being performed, so that there is no ambiguity.
    if data in ['sider.csv', 'clintox.csv']:
        task_type = 'Multi-label Classification'

    elif data in ['BBBP.csv', 'bace.csv']:
        task_type = 'Single-label Classification'

    elif data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv']:
        task_type = 'Regression'




    return checker, task_type