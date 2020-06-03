from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def classifier(model):
    """Returns model specific classifier function."""

    # Create Dictionary of classifiers to be called with self.algorithm as key.

    classifiers = {
        'svc': SVC,
        'knc': KNeighborsClassifier,
        'rfc': RandomForestClassifier
    }
    return classifiers[model]