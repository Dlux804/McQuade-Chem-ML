# This function allows for main.py to skip over algorithm/data set combinations that are not compatible.

def Get_Task_Type_1(data, alg):
    if data in ['sider.csv', 'clintox.csv'] and alg == 'svc':
        checker = 0
    elif data in ['BBBP.csv', 'sider.csv', 'clintox.csv', 'bace.csv'] and alg in ['ada', 'svr', 'gdb', 'nn',
                                                                                  'knn']:
        checker = 0
    elif data in ['ESOL.csv', 'Lipophilicity-ID.csv', 'water-energy.csv', 'logP14k.csv', 'jak2_pic50.csv'] and alg in [
        'svc', 'knc']:
        checker = 0

    else:
        checker = 1

    return checker