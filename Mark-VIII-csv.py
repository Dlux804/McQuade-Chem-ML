# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:24:43 2019

@author: quang

Objective: Make robust classes that can read a csv, generate rdkit features and ecfp and do machine learning on every sklearn model
## TLDR: Let's pretend that I'm useful and get paid the big bucks at the same time!!!!!! 
    
    Mark-V now has rf, svr, gdb, adaboost, MLP (neural net) and knn.

    Things to include in Mark-VI:
        Some foolproof for typos.
        An option to use old optimized parameters for large iterations. Need to run hyperparam on Onix to get the best params.
        

"""

#Import Packages. Let's try to be as efficient as possible
# =============================================================================
from itertools import combinations 
import pandas as pd
import numpy as np
from rdkit import Chem # potentially unecessary, but solved problem of not being able to find a call
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from time import time
# =============================================================================

exp = input('Experiment Number (3 digits): ')
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


log = pd.DataFrame(columns = ['Date','Features', 'Model Name','Parameter Grid', 'Opt. Hyperparameters','Training Percent', 'Featurization Time (sec)','Training Time (sec)', 'Tuning Time (sec)','R^2', 'MSE', 'RMSE'])
log.at[exp,'Date'] = dt_string

class csvhandling:
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        #Find smiles automatically from a csv
    def findsmiles(self):
        csv = pd.read_csv(self)
        for i in csv.head(0):
            try:
                pd.DataFrame(list(map(Chem.MolFromSmiles, csv[i])))
                smiles_col = csv[i]
#                molob_col = pd.DataFrame(molob, columns = 'molobj')
            except TypeError:
                pass
        return csv, smiles_col
      

#Feature selection 
def feature_select(csv_file, model_name, selected_feat = None):
    df, smiles_col = csvhandling.findsmiles(csv_file)
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts', 'morganchiral3counts', 'atompaircounts']
    log.at[exp,'Model Name'] = model_name
    if model_name == 'nn' or model_name == 'knn':
        feat_sets.remove('rdkit2d')
        print(feat_sets)
        if selected_feat == None: #ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
            selected_feat = [int(x) for x in input('Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
           
        selected_feat = [feat_sets[i] for i in selected_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')
    else:
        if selected_feat == None: #ask for features
            print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
            [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
            # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
            selected_feat = [int(x) for x in input('Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]
        selected_feat = [feat_sets[i] for i in selected_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')
        
        
    return selected_feat




class param:
    def Adaparamgrid():
        base_estimator = [tree.DecisionTreeRegressor(max_features='sqrt', splitter='best', max_depth=3), GradientBoostingRegressor(), SVR(kernel = 'linear'),RandomForestRegressor(n_estimators=500)]
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 30)]
        learning_rate = [0.001,0.005,0.01,0.05,0.1,0.5,1]
        param_grid = {'base_estimator': base_estimator,
                      'n_estimators': n_estimators,
                      'learning_rate': learning_rate}
        return param_grid
    def RFparamgrid():
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)] #Number of trees
        max_features = ['auto', 'sqrt']     # Number of features to consider at every split
        max_depth = [int(x) for x in np.linspace(1, 30, num = 11)] # Maximum number of levels in tree
        min_samples_split = [2, 4 ,6 ,8, 10] # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2,3, 4,5,6]# Minimum number of samples required at each leaf node
        bootstrap = [True, False]# Method of selecting samples for training each tree
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
        return param_grid
    def SVRparamgrid():
    #Kernel functions
        kernel = ['rbf', 'poly', 'linear']
    #Penalty parameter C of the error term.
        Cs = [0.001, 0.005 ,0.01, 0.05 ,0.1, 0.5, 1, 5,10,100]
    #epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
        epsilon = [0.1,0.2,0.3,0.4, 0.5,0.6]
    #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        gammas = [0.001, 0.005 ,0.01, 0.05 ,0.1, 0.5, 1]
    #Degree of the polynomial kernel function (‘poly’)
        degrees = [1,2,3,4,5]
        param_grid = {'kernel': kernel,'C': Cs, 'gamma' : gammas, 'epsilon': epsilon, 'degree' : degrees}
        return param_grid
    def GDBparamgrid():
    #Number of trees
        n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 20)]
    # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 25, num = 24, endpoint=True)]
    # Minimum number of samples required to split a node
        min_samples_split = [int(x) for x in np.linspace(2, 30, num = 10, endpoint=True)]
    # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(2, 30, num = 10, endpoint=True)]
    #learning rate
        learning_rate = [0.001,0.005,0.01,0.05,0.1,0.5,1]
        param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}
        return param_grid
    def MLPparamgrid():
        #Number of hidden layers
        hidden_layer_sizes = [(100,), (100,50,100), (50,100,50), (np.random.randint(low = 50, high = 100, size = 10))]
        #Activation function for the hidden layer.
        activation = ['tanh', 'logistic']
        #The solver for weight optimization.
        solver = ['adam', 'sgd']
        #L2 penalty (regularization term) parameter.
        alpha = [0.0001, 0.0005,0.001,0.005,0.01,0.05,0.1]
        #Learning rate 
        learning_rate = ['constant','adaptive', 'invscaling']
        param_grid= {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'alpha': alpha,
                'learning_rate': learning_rate,
                }
        return param_grid
    def KNNparamgrid():
        #Number of neighbors to use
        n_neighbors = [int(x) for x in np.linspace(start = 5, stop = 30, num = 20)]
        #weight function used in prediction
        weights = ['uniform', 'distance']
        #Algorithm used to compute the nearest neighbors:
        algorithm = ['auto', 'ball_tree', 'kd_tree']
        #Leaf size passed to BallTree or KDTree
        leaf_size = [int(x) for x in np.linspace(start = 20, stop = 50, num = 15)]
        #Power parameter for the Minkowski metric
        p = [1,2,3,4,5]
        param_grid = { 'n_neighbors': n_neighbors,
                      'weights': weights,
                      'algorithm': algorithm,
                      'leaf_size': leaf_size,
                      'p': p
                }
        return param_grid


#Run ML for non hyperparameter
def predict(regressor, train_features, test_features, train_target, test_target):
    fit_time = time()
    regressor.fit(train_features, train_target)
    done_time = time()
    fit_time = done_time - fit_time
    print('Finished Training After ',(done_time-fit_time),"sec\n")
    # Make predictions
    predictions = regressor.predict(test_features)  # Val predictions

    true = test_target
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = true
    pva['predicted'] = predictions
#    pva.to_csv(exp+expt+'-pva_data.csv')

    return pva, fit_time
    
#Hyper parameter tunes using RandomizedSearchCV
def hyperTune(model,model_name, train_features, train_target, grid, folds, iters, expt):
    print("Starting Hyperparameter tuning\n")
    start_tune = time()
    search_random = RandomizedSearchCV(estimator= model, param_distributions=grid, n_iter=iters, cv=folds, verbose=2,
                                   random_state=42, n_jobs=4)
    log.at[exp, 'Parameter Grid'] = grid
    # Fit the random search model
    search_random.fit(train_features, train_target)
    tuned = search_random.best_params_
    param_dict = {str(model_name)+'-'+expt: tuned}
    
    stop_tune = time()
    time_par = {str(model_name)+'-'+expt: (stop_tune-start_tune)}
    print('Best Parameter Found After ',(stop_tune-start_tune),"sec\n")
    print(tuned)
    return tuned, param_dict, time_par

#Graphs for ML results
def pva_graphs(pva,model_name, expt):
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
    print('R^2 = %.3f' % r2)
    print('MSE = %.3f' % mse)
    print('RMSE = %.3f' % rmse)
    r2_dict = {expt +' R^2': r2}
    mse_dict= {expt +' MSE': mse}
    rmse_dict = {expt + ' RMSE': rmse}
    plt.rcParams['figure.figsize']= [15,9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    plt.plot(pva['actual'], pva['predicted'], 'o')
    # ax = plt.axes()
    plt.xlabel('True', fontsize=14);
    plt.ylabel('Predicted', fontsize=14);
    plt.title('EXP:'+exp+expt)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]
    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    plt.savefig(model_name+'-'+exp+'-'+expt +'.png')
    plt.show()
    return r2_dict, mse_dict, rmse_dict

def runRegress(model, model_name,train_features, test_features, train_target, test_target, expt):
    start_reg = time()
    print("Starting " + model_name + expt +'\n')
    pva, time_var = defaultML(model,model_name, train_features, test_features, train_target, test_target, expt)
    r2_dict, mse_dict, rmse_dict = pva_graphs(pva,model_name, expt)
    stop_reg = time()
    print(model_name + expt+ ' Finished after ',(stop_reg-start_reg),"sec\n")
    return r2_dict, mse_dict, rmse_dict, time_var
#MRun every single ML algorithm with and without hyperparameter
def RegressNorm(train_features, test_features, train_target, test_target, csv_file, target_colname ,model_name, expt):
    
    if model_name == 'rf':
        model = RandomForestRegressor()
    elif model_name == 'svr':
        model = SVR()
    elif model_name == 'gdb':
        model = GradientBoostingRegressor()
    elif model_name == 'ada':
        model = AdaBoostRegressor()
    elif model_name == 'nn':
        model = MLPRegressor()
    elif model_name == 'knn':
        model = KNeighborsRegressor()
    return runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)

def RegressHyper(train_features, test_features, train_target, test_target, csv_file, target_colname ,model_name, folds, iters, expt):
    if model_name == 'rf':
        model = RandomForestRegressor()
        params,param_dict, time_par = hyperTune(model,model_name, train_features, train_target, param.RFparamgrid(),  folds, iters, expt)
        model = RandomForestRegressor(n_estimators = params['n_estimators'], max_features = params['max_features'],
                           max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                           bootstrap = params['bootstrap'], min_samples_leaf = params['min_samples_leaf'],
                           random_state = 25)
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)
    elif model_name == 'svr':
        model = SVR()
        params, param_dict,time_par = hyperTune(model,model_name, train_features, train_target,param.SVRparamgrid(),  folds, iters, expt)
        model = SVR(kernel = params['kernel'], C = params['C'], gamma = params['gamma'], epsilon = params['epsilon'], degree = params['degree'])
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)


    elif model_name == 'gdb':
        model = GradientBoostingRegressor()
        params,param_dict, time_par = hyperTune(model,model_name, train_features, train_target,param.GDBparamgrid(), folds, iters, expt)
        model = GradientBoostingRegressor(n_estimators=params['n_estimators'], max_features=params['max_features'],
                               max_depth=params['max_depth'], min_samples_split=params['min_samples_split']
                               , min_samples_leaf=params['min_samples_leaf'],learning_rate = params['learning_rate'],
                               random_state=25)
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)


    elif model_name == 'ada':
        model = AdaBoostRegressor()
        params,param_dict, time_par = hyperTune(model,model_name, train_features, train_target,param.Adaparamgrid(), folds, iters, expt)
        model = AdaBoostRegressor(base_estimator= params['base_estimator'], n_estimators = params['n_estimators'], learning_rate= params['learning_rate'], random_state=25)
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)
    elif model_name == 'nn':
        model = MLPRegressor()
        params,param_dict, time_par = hyperTune(model,model_name, train_features, train_target,param.MLPparamgrid(), folds, iters, expt)
        model = MLPRegressor(hidden_layer_sizes = params['hidden_layer_sizes'], activation = params['activation'], solver = params['solver'], 
                         alpha = params['alpha'], learning_rate = params['learning_rate'], random_state = 25)
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)
    elif model_name == 'knn':
        model = KNeighborsRegressor()
        params, param_dict,time_par = hyperTune(model,model_name, train_features, train_target,param.KNNparamgrid(), folds, iters, expt)
        model = KNeighborsRegressor(n_neighbors= params['n_neighbors'], weights = params['weights'], algorithm= params['algorithm'], leaf_size = params['leaf_size'], p = params['p'])
        r2_dict, mse_dict, rmse_dict, time_var = runRegress(model, model_name,train_features, test_features, train_target, test_target, expt)

    return r2_dict, mse_dict, rmse_dict,param_dict, time_var, time_par
    

class RunML:
    def __init__(self, csv_file, target_colname,model_name):
        self.csv_file = csv_file
        self.target_colname = target_colname
        self.model_name = model_name
    def Default(csv_file, target_colname,model_name, selected_feat = None):
        df, smiles_col, feat_sets = feature_select(csv_file, model_name, selected_feat)
        feature_lst = []
        train_lst = []
        r2_lst = []
        mse_lst = []
        rmse_lst = []
        feat_time = []
        feat_name = []
        index = [1,2]
        for i in index:
            combina = list(combinations(feat_sets, i))
            for comb in combina:
                expt = str(comb)
                feat_name.append(comb)
                log.at[exp,'Features'] = feat_name
                generator = MakeGenerator(comb)
                start_feat = time()
                make_features = list(map(generator.process, smiles_col)) #Make a list of features from SMILES
                stop_feat = time()
                time_feat = {str(model_name)+ '-' + expt: (stop_feat-start_feat)}
                feat_time.append(time_feat)
                log.at[exp,'Featurization Time (sec)'] = feat_time
                feature_lst.append(make_features)
                for i in feature_lst:
                    featuresdf = pd.DataFrame(i)
                    featuresdf = featuresdf.dropna() #Dataframe of only features
                    featuresarr = np.array(featuresdf)
                    train_features, test_features, train_target, test_target = train_test_split(featuresarr, df[target_colname], test_size = 0.2, random_state = 42)
                r2_dict, mse_dict, rmse_dict , train_time = RegressNorm(train_features, test_features, train_target, test_target, csv_file, target_colname, model_name, expt)
                r2_lst.append(r2_dict)
                mse_lst.append(mse_dict)
                rmse_lst.append(rmse_dict)
                log.at[exp,'R^2'] = r2_lst
                log.at[exp, 'MSE'] = mse_lst
                log.at[exp, 'RMSE'] = rmse_lst
                train_lst.append(train_time)
                log.at[exp,'Training Time (sec)'] = train_lst
                log.at[exp, 'Parameter Grid'] = 'None'
                log.at[exp, 'Opt. Hyperparameters'] = 'None'
                log.at[exp,'Tuning Time (sec)'] = 'None'
                log.at[exp,'Training Percent'] = 0.8
                log.to_csv(exp+'-results.csv')
        return r2_dict, mse_dict, rmse_dict
    def Hyperparam(csv_file, target_colname,model_name, selected_feat = None):
        folds = int(input('Please state the number of folds for hyperparameter searching: '))
        iters = int(input('Please state the number of iterations for hyperparameter searching: '))
        df, smiles_col, feat_sets = feature_select(csv_file, model_name, selected_feat)
        feature_lst = []
        train_lst = []
        hyper_lst = []
        r2_lst = []
        mse_lst = []
        rmse_lst = []
        param_lst = []
        feat_time = []
        feat_name = []
        index = [1,2]
        for i in index:
            combina = list(combinations(feat_sets, i))
            for comb in combina:
                expt = str(comb)
                feat_name.append(comb)
                log.at[exp,'Features'] = feat_name
                generator = MakeGenerator(comb)
                start_feat = time()
                make_features = list(map(generator.process, smiles_col)) #Make a list of features from SMILES
                stop_feat = time()
                time_feat = {str(model_name)+ '-' + expt: (stop_feat-start_feat)}
                feat_time.append(time_feat)
                log.at[exp,'Featurization Time (sec)'] = feat_time
                feature_lst.append(make_features)
                for i in feature_lst:
                    featuresdf = pd.DataFrame(i)
                    featuresdf = featuresdf.dropna() #Dataframe of only features
                    featuresarr = np.array(featuresdf)
                    train_features, test_features, train_target, test_target = train_test_split(featuresarr, df[target_colname], test_size = 0.2, random_state = 42)
                r2_dict, mse_dict, rmse_dict ,param_dict, train_time, hyper_time = RegressHyper(train_features, test_features, train_target, test_target, csv_file, target_colname, model_name, folds, iters,expt)
                param_lst.append(param_dict)
                log.at[exp, 'Opt. Hyperparameters'] = param_lst
                r2_lst.append(r2_dict)
                mse_lst.append(mse_dict)
                rmse_lst.append(rmse_dict)
                log.at[exp,'R^2'] = r2_lst
                log.at[exp, 'MSE'] = mse_lst
                log.at[exp, 'RMSE'] = rmse_lst
                train_lst.append(train_time)
                log.at[exp,'Training Time (sec)'] = train_lst
                hyper_lst.append(hyper_time)
                log.at[exp,'Tuning Time (sec)'] = hyper_lst
                log.at[exp,'Training Percent'] = 0.8
                log.to_csv(exp+'-results.csv')

##                total_time.append(time_lst)
        return r2_dict, mse_dict, rmse_dict
    
RunML.Hyperparam('dataFiles/water-energy.csv', 'expt', 'ada')
