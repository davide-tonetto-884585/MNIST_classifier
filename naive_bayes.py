import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, check_array


class BetaNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=0.05):
        super().__init__()
        self.epsilon = epsilon

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y_ = y.to_numpy() if isinstance(y, pd.DataFrame) else y
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = self.X_.shape[1]
        self.n_samples_ = self.X_.shape[0]
        self.prior_ = np.bincount(self.y_) / self.n_samples_

        # compute alpha and beta for each class
        self.alpha_ = []
        self.beta_ = []
        for i, c in enumerate(self.classes_):
            X_c = self.X_[self.y_ == c]
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0)

            # compute alpha and beta, avoid division by zero
            var[var == 0] = 1e-6

            k = ((mean * (1 - mean)) / var) - 1
            alpha = mean * k
            beta = (1 - mean) * k

            # set alpha and beta to 0 if they are negative
            alpha[alpha < 0] = 1e-6
            beta[beta < 0] = 1e-6

            # store alpha and beta
            self.alpha_.append(alpha)
            self.beta_.append(beta)

        return self

    def predict(self, X):
        # check is fit had been called
        check_is_fitted(self)

        # Input validation
        check_array(X)

        # transform X into numpy array
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

        # compute the probability of each class for each sample
        y_pred = []
        for x in X:
            prob = []
            for i, c in enumerate(self.classes_):
                probs = beta.cdf(x + self.epsilon, self.alpha_[i], self.beta_[i]) - beta.cdf(x - self.epsilon,
                                                                                             self.alpha_[i],
                                                                                             self.beta_[i])
                np.nan_to_num(probs, copy=False, nan=1.0)
                prob.append(np.prod(probs) * self.prior_[i])

            # get the class with the highest probability
            y_pred.append(np.argmax(prob, axis=0))

        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_mean_images(self):
        means = []
        for i, c in enumerate(self.classes_):
            mean = self.alpha_[i] / (self.alpha_[i] + self.beta_[i])
            means.append(mean.reshape(28, 28))

        return means
