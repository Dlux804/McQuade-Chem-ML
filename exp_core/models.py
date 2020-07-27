"""
Objective: Initialize Machine Learning pipeline
"""
import os
from exp_core import misc
from main import ROOT_DIR
os.chdir(ROOT_DIR)  # Start in root directory
from exp_core.estimator import Estimator


class MLModel(Estimator):
    def __init__(self, algorithm, dataset, target, feat_meth, tune, cv_folds, opt_iter):
        Estimator.__init__(self, algorithm, dataset, target, feat_meth, tune, cv_folds, opt_iter)


with misc.cd('dataFiles'):  # Initialize model
    model1 = MLModel(algorithm="gdb", dataset="ESOL.csv", target="water-sol", feat_meth=[0], tune=True, cv_folds=2,
                     opt_iter=2)
    model1.load_smiles()
    model1.name()
    model1.featurize()
    model1.data_split()
    print(model1.feature_list)
    model1.get_regressor()
    model1.make_grid()
    model1.hyperTune()
