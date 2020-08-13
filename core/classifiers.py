from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def get_classifier(self, call=False):
    """Returns model specific classifier function."""

    # Create Dictionary of classifiers to be called with self.algorithm as key.

    skl_cls = {
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'rf': RandomForestClassifier,
        'linearSVC': LinearSVC,
        'ada': AdaBoostClassifier,
        'gdb': GradientBoostingClassifier
    }
    # if self.algorithm in skl_cls.keys():
    #     self.estimator = skl_cls[self.algorithm]()
    # else:
    #     pass

    if call:
        self.estimator = skl_cls[self.algorithm]

    # return instance with either default or tuned params
    else:
        if hasattr(self, 'params'):  # has been tuned
            self.estimator = skl_cls[self.algorithm](**self.params)
        else:  # use default params
            self.estimator = skl_cls[self.algorithm]()
