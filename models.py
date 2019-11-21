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


    # featurize = features.feature_select


# Initiate Model
model1 = MlModel('rf', 'ESOL.csv', 'water-sol')

# Featurize molecules and add to class instance
# model1.data, model1.features = features.featurize(model1.data, model1.algorithm, [0])
model1.featurization([0])
what = model1.data

print(what)
print(model1.feat_meth)