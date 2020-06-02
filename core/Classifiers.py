from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def classifier(model):
    """Returns model specific classifier function."""

    # Create Dictionary of classifiers to be called with self.algorithm as key.

    classifiers = {
        'svc': SVC,
        'knc': KNeighborsClassifier,
    }
    return classifiers[model]