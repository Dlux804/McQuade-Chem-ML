"""
Objective: The goal of this script is to create py2neo's graph classes so that we can use the model instance to create
graphs in Neo4j
"""

# from py2neo import Graph, Relationship
from py2neo.ogm import GraphObject, Property, RelatedFrom, RelatedTo, Label


class TuningAlg(GraphObject):
    algorithm = Property()
    num_cv = Property()
    tuneTime = Property()
    delta = Property()
    n_best = Property()
    steps = Property()
    Uses_Tuning = RelatedFrom("Algorithm")


class FeatureList(GraphObject):
    num = Property()


class TrainingSet(GraphObject):
    size = Property()
    Splits_Into = RelatedFrom("RandomSplit")
    Contains = RelatedTo("Molecules")


class MlModel(GraphObject):
    train_time = Property()
    date = Property()
    feat_time = Property()
    test_time = Property()
    Has_Features = RelatedTo("FeatureList")
    Trains = RelatedTo("TrainingSet")
    Uses_Algorithm = RelatedTo("Algorithm")
    Uses = RelatedTo("Dataset")
    Predicts = RelatedTo("TestSet")
    Uses_Featurization = RelatedTo("Features")


class Dataset(GraphObject):
    source = Property()
    data = Property()
    measurement = Property()
    Contains_Molecules = RelatedTo("Molecules")


class Algorithm(GraphObject):
    source = Property()
    tuned = Property()
    name = Property()
    version = Property()


class RandomSplit(GraphObject):
    test_percent = Property()
    train_percent = Property()
    random_seed = Property()
    Splits_Into = RelatedTo("TestSet")


class TestSet(GraphObject):
    RMSE = Property()
    MSE = Property()
    R2 = Property()
    size = Property()


class Features(GraphObject):
    name = Property()

    def __init__(self, feature_name):
        feature_name = Label(''.join(feature_name))


class Molecules(GraphObject):
    SMILES = Property()
    target = Property()


