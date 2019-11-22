'''
One way to handle the OOP of this code base is to handle each model as the instance of a class.
Required attributes will be what type of model and what data it uses.
Defaults will handle the splitting type, tuning etc.

Results of the model will be stored in the class instance,
for example you could have a model.pva be the pva graph and
model.results be the r2, rmse, time, etc.
'''
import ingest
import features
import grid
import regressors
class MlModel:
    def __init__(self, algorithm, dataset, target):
        """Learning algorithm, dataset and target property's column name."""
        self.algorithm = algorithm
        self.dataset = dataset
        self.target = target
        self.data, self.smiles = ingest.load_smiles(self,dataset)

    def featurization(self, feats=None):
        """ Featurizes molecules in dataset.
            Keyword arguments:
            feats -- Features you want.  Default = None (requires user input)
        """
        self.data, self.feat_meth = features.featurize(self.data, self.algorithm, feats)

    def run(self, tune=False):
        """ Runs model. Returns log of results and graphs."""
        # Split data up
        train_features, test_features, train_target, test_target, self.feature_list = features.targets_features(self.data)
        self.regressor = regressors.regressor(self.algorithm)
        if tune:  # Do hyperparameter tuning

            # ask for tuning variables
            folds = int(input('Please state the number of folds for hyperparameter searching: '))
            iters = int(input('Please state the number of iterations for hyperparameter searching: '))

            # Make parameter grid
            param_grid = grid.make_grid(self.algorithm)
            params, param_dict, tuneTime = regressors.hyperTune(self.regressor, self.algorithm, train_features,
                                                                train_target, param_grid, folds, iters, self.feat_meth)








    # featurize = features.feature_select


# Initiate Model
model1 = MlModel('rf', 'ESOL.csv', 'water-sol')

# Featurize molecules and add to class instance
# model1.data, model1.features = features.featurize(model1.data, model1.algorithm, [0])
model1.featurization([0])
what = model1.data

print(what)
print(model1.feat_meth)