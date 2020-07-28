"""
Objective: Receive ML data input
"""


class MLModel:
    def __init__(self, algorithm, dataset, target, feat_meth, tune, cv_folds, opt_iter):

        self.algorithm = algorithm
        self.dataset = dataset
        self.target = target
        self.feat_meth = feat_meth
        self.tuned = tune
        self.cv = cv_folds
        self.opt_iter = opt_iter


