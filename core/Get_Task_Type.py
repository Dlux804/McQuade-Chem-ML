""" This function allows for main.py to skip over algorithm/data set combinations that are not compatible.

The checker variable is set to 0 if the combination is not compatible, and 1 otherwise.

It is then returned to main.py, where it is used to decide if the model/data set combination should be ran,
or the combination should be skipped.

"""
def Get_Task_Type_1(data, alg):
    if data in ['sider.csv', 'clintox.csv'] and alg == 'svc':  # SVC is not compatible with multi-label classification.
        checker = 0

        # These classification data sets are not compatible with these regression models
    elif data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv'] and alg in ['ada', 'svr', 'gdb', 'nn']:
        checker = 0

        # These regression data sets are not compatible with these classification models.
    elif data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv'] and alg in [
        'svc']:
        checker = 0

    else: # Checker = 1 when there are no compatibility
        checker = 1

    return checker