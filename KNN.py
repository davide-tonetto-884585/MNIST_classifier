from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, euclidean_distances
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import pandas as pd


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y_ = y.to_numpy() if isinstance(y, pd.DataFrame) else y

        # n features in
        self.n_features_in_ = self.X_.shape[1]

        return self

    def predict(self, X):
        # check is fit had been called
        check_is_fitted(self)

        # Input validation
        check_array(X)

        # transform X into numpy array
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

        closest = np.argpartition(euclidean_distances(X, self.X_), axis=1, kth=self.k)[:, :self.k]

        # foreach k-ple of nearest neighbors, get the most frequent label
        y_pred = np.apply_along_axis(lambda x: np.bincount(self.y_[x]).argmax(), axis=1, arr=closest)

        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
