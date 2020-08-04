from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_classifier(self):
    """Returns model specific classifier function."""

    # Create Dictionary of classifiers to be called with self.algorithm as key.

    skl_cls = {
        'svc': SVC,
        'knc': KNeighborsClassifier,
        'rf': RandomForestClassifier
    }
    if self.algorithm in skl_cls.keys():
        self.estimator = skl_cls[self.algorithm]()
    else:
        pass

