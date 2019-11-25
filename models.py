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
import analysis

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
        # Split data up. Set random seed here for graph comparison purposes.
        train_features, test_features, train_target, test_target, self.feature_list = features.targets_features(self.data, self.target, random=42)

        # set the model specific regressor function from sklearn
        self.regressor = regressors.regressor(self.algorithm)

        if tune:  # Do hyperparameter tuning

            # ask for tuning variables
            folds = int(input('Please state the number of folds for hyperparameter searching: '))
            iters = int(input('Please state the number of iterations for hyperparameter searching: '))
            jobs = int(input('Input the number of processing cores to use. (-1) to use all.'))

            # Make parameter grid
            param_grid = grid.make_grid(self.algorithm)

            # Run Hyper Tuning
            params,  tuneTime = regressors.hyperTune(self.regressor(), train_features,
                                                                train_target, param_grid, folds, iters, jobs=jobs)

            # redefine regressor model with best parameters.
            self.regressor = self.regressor(**params)  # **dict will unpack a dictionary for use as keywrdargs

        # Done tuning, time to fit and predict
        pva, fit_time = analysis.predict(self.regressor(), train_features, test_features, train_target, test_target)
        self.graph = analysis.pva_graphs(pva, self.algorithm)

        # run the model 5 times and collect the metric stats as dictionary
        self.stats = analysis.replicate_model(self, 5)



# Initiate Model
model1 = MlModel('rf', 'water-energy.csv', 'expt')

# featurize data with rdkit2d
model1.featurization([0])
print(model1.feat_meth)

# isolate dataframe
what = model1.data
print(what)

# Run the model with hyperparameter optimization
model1.run(tune=False)

# display PvA graph
model1.graph.show()

# model statistics
print(model1.stats)



