from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classifier(model):
    """Returns model specific classifier function."""

    # Create Dictionary of classifiers to be called with self.algorithm as key.

    skl_cls = {
        'svc': SVC,
        'knc': KNeighborsClassifier,
        'rf': RandomForestClassifier
    }
    if model.algorithm in skl_cls.keys():
        regressor = skl_cls[model.algorithm]
        # TODO refactor self.regressor to something more general (learner? method? algorithm? ESTIMATOR)
        task_type = 'classification'
        return regressor, task_type
