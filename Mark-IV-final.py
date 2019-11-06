# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:24:43 2019

@author: quang

Objective: Make robust classes that can read a csv, generate rdkit features and ecfp and do machine learning on every sklearn model
## TLDR: Let's pretend that I'm useful and get paid the big bucks at the same time!!!!!! 

    Things to include in Mark-IV:
        Include Neural Net, Adaboost and K-nearest neighbors.
        Some foolproof for typos.
        An option to use old optimized parameters for large iterations
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
# =============================================================================
import timeit
start=timeit.default_timer()
exp = input('Experiment Number (3 digits):')
#graph_title = input('Graph Name:')
#image_name= input('Name of the image:')
#This class is for reading csv into DF, find the smiles column by itself and creat a molobject column.
def featurelist():
    feature_list = []
    generator = MakeGenerator(("RDKit2D",))
    for name, numpy_type in generator.GetColumns():
        feature_list.append(name) #Features's headers
    return feature_list   

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
def feature_select(csv_file, selected_feat = None):
    df, smiles_col = csvhandling.findsmiles(csv_file)
    feat_sets = ['rdkit2d', 'rdkit2dnormalized', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts', 'morganchiral3counts', 'atompaircounts']

    if selected_feat == None: #ask for features
        print('   {:5}    {:>15}'.format("Selection", "Featurization Method"))
        [print('{:^15} {}'.format(*feat)) for feat in enumerate(feat_sets)];
        # selected_feat = input('Choose your features from list above.  You can choose multiple with \'space\' delimiter')
        selected_feat = [int(x) for x in input('Choose your features  by number from list above.  You can choose multiple with \'space\' delimiter:  ').split()]

        selected_feat = [feat_sets[i] for i in selected_feat]
        print("You have selected the following featurizations: ", end="   ", flush=True)
        print(*selected_feat, sep=', ')
    elif selected_feat == 'all':
        selected_feat = feat_sets
        
        
    return df, smiles_col ,selected_feat




class param:
    def Adaparamgrid():
        base_estimator = [tree.DecisionTreeRegressor(max_features='sqrt', splitter='best', max_depth=3), GradientBoostingRegressor(), RandomForestRegressor(n_estimators=500)]
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 40)]
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

#Run ML for non hyperparameter
def defaultML(model, train_features, test_features, train_target, test_target, expt):
    model.fit(train_features, train_target)

    # Make predictions
    predictions = model.predict(test_features)  # Val predictions

    # predictions
    predictions = model.predict(test_features)
    true = test_target
    pva = pd.DataFrame([], columns=['actual', 'predicted'])
    pva['actual'] = true
    pva['predicted'] = predictions
    # print(pva)
    pva.to_csv(exp+expt+'-pva_data.csv')
#    pva.to_csv(exp + '-pva_data.csv')

    return pva

#Hyper parameter tunes using RandomizedSearchCV
def hyperTune(model, train_features, train_target, grid, folds, iters):
    
    search_random = RandomizedSearchCV(estimator= model, param_distributions=grid, n_iter=iters, cv=folds, verbose=1,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    search_random.fit(train_features, train_target)
    tuned = search_random.best_params_
    print(tuned)
    return tuned

#Graphs for ML results
def pva_graphs(pva, expt):
    r2 = r2_score(pva['actual'], pva['predicted'])
    mse = mean_squared_error(pva['actual'], pva['predicted'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
    print('R^2 = %.3f' % r2)
    print('MSE = %.3f' % mse)
    print('RMSE = %.3f' % rmse)
    
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

    plt.savefig(exp+expt +'.png')
    plt.show()

    return r2, mse, rmse
#MRun every single ML algorithm with and without hyperparameter
def Regression(train_features, test_features, train_target, test_target, csv_file, target_colname ,model_name, expt, paramsearch = False):
    
#    train_features, test_features, train_target, test_target = splitforml(csv_file, target_colname)

    if not (paramsearch):
        if model_name == 'rf':
            print('You better pay me for this')
            model = RandomForestRegressor()
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
    
        elif model_name == 'svr':
            print('You better pay me for this')
            model = SVR()
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
        
        elif model_name == 'gdb':
            print('You better pay me for this')
            model = GradientBoostingRegressor()
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
        elif model_name == 'ada':
            print('You better pay me for this')
            model = AdaBoostRegressor()
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
    else:
        folds = int(input('Please state the number of folds for hyperparameter searching: '))
        iters = int(input('Please state the number of iterations for hyperparameter searching: '))
        if model_name == 'rf':
            print('You better pay me for this')
            model = RandomForestRegressor()
            params = hyperTune(model, train_features, train_target, param.RFparamgrid(),  folds, iters)
            model = RandomForestRegressor(n_estimators = params['n_estimators'], max_features = params['max_features'],
                           max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                           bootstrap = params['bootstrap'], min_samples_leaf = params['min_samples_leaf'],
                           random_state = 25)
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
        elif model_name == 'svr':
            print('You better pay me for this')
            model = SVR()
            params = hyperTune(model, train_features, train_target,param.SVRparamgrid(),  folds, iters)
            model = SVR(kernel = params['kernel'], C = params['C'], gamma = params['gamma'], epsilon = params['epsilon'], degree = params['degree'])
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
        elif model_name == 'gdb':
            print('You better pay me for this')
            model = GradientBoostingRegressor()
            params = hyperTune(model, train_features, train_target,param.GDBparamgrid(), folds, iters)
            model = GradientBoostingRegressor(n_estimators=params['n_estimators'], max_features=params['max_features'],
                               max_depth=params['max_depth'], min_samples_split=params['min_samples_split']
                               , min_samples_leaf=params['min_samples_leaf'],learning_rate = params['learning_rate'],
                               random_state=25)
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
        elif model_name == 'ada':
            print('You better pay me for this')
            model = AdaBoostRegressor()
            params = hyperTune(model, train_features, train_target,param.Adaparamgrid(), folds, iters)
            model = AdaBoostRegressor(base_estimator= params['base_estimator'], n_estimators = params['n_estimators'], learning_rate= params['learning_rate'], random_state=25)
            pva = defaultML(model, train_features, test_features, train_target, test_target, expt)
            graph = pva_graphs(pva, expt)
    return graph   
    

#Regression('water-energy.csv', 'expt', 'rf')
def Allcombo(csv_file, target_colname,model_name, selected_feat = None, paramsearch = False ):
    """
    Listen up kids, I won't go over this again. I SAID LISTEN UP!!!!
    Function: This function allows you to run machine learning on all possible feature combinations by themselves and in pairs.
    
    Instructions for variable inputs:
    csv_file: string. Name of the csv file that has all of your shit. INCLUDE ".CSV" !!!
    target_column: string. This is the name(header) of the label/target column that you're trying to predict.
    model_name: string. Model of choice :'rf', 'svg', 'xgb'.
    paramsearch = False (default). If not mentioned or inputed as False, model will not do hyper parameter searching.
                                    If 'True', then you will be asked for number of folds and iterations. 
    
    Output: graphs and csv files for each of the combination
    Example: Allcombo('water-energy.csv', 'expt', 'ada',None, True)
                csv_file: The name of my csv file is 'Lipophilicity-main.csv'
                'exp': 'exp' is the target column's name for the lipo data. 
                'model_name': rf is short for Random Forest 
                selected_feat: If selected_feat = None, then user input will be prompted. You can also manually in put a list of features that range of 0 to 6 (Ex: [0,3] means feature 0 and 3)
                                You can also set selected_feat = 'all', this will make every feature combination either by themselves or in pairs.
                                                                                                                     )
                'paramsearch': If leave blank or False, then it will not run hyperparam. If input it True, then it will ask for number of folds and iterations.
    
    Advantage: Allow you to run ML of your choice, with or without hyperparam with all possible combinations of your feature of choice. 
                It will also name every graph and files according to the feature combination and experiment number
    Things to come: Neural Net, K-nearest neighbor. 
    Things to improve: 
            Typo proof
            How to effiently normaliza data for NN and KNN
            Still need to think of a way to efficiently store hyperparams
            Remove the True columns on every descriptors.
    """
    
    
    df, smiles_col, feat_sets = feature_select(csv_file, selected_feat)
    feature_lst = []
    index = [1,2]
    for i in index:
        combina = list(combinations(feat_sets, i))
        for comb in combina:
            expt = str(comb)
            generator = MakeGenerator(comb)
            make_features = list(map(generator.process, smiles_col)) #Make a list of features from SMILES
            feature_lst.append(make_features)
            for i in feature_lst:
                featuresdf = pd.DataFrame(i)
                featuresdf = featuresdf.dropna() #Dataframe of only features
                featuresarr = np.array(featuresdf)
                train_features, test_features, train_target, test_target = train_test_split(featuresarr, df[target_colname], test_size = 0.2, random_state = 42)
            if not (paramsearch):
                regres = Regression(train_features, test_features, train_target, test_target, csv_file, target_colname, model_name, expt)
            else:
                regres = Regression(train_features, test_features, train_target, test_target, csv_file, target_colname, model_name, expt, True)
    return regres




Allcombo('water-energy.csv', 'expt', 'ada',None, True)
stop = timeit.default_timer()
print('Finished after ',(stop-start),"sec")