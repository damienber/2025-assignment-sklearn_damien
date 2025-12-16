"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement:
- a scikit-learn estimator for KNearestNeighbors classification.
- a scikit-learn cross-validation splitter based on monthly time periods.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors classifier.

    A simple k-nearest neighbors classifier using Euclidean distance.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the KNearestNeighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use for prediction.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        X, y = validate_data(self, X=X, y=y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict class labels for the provided data."""
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        distances = pairwise_distances(X, self.X_)
        neigh_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neigh_labels = self.y_[neigh_indices]

        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i in range(X.shape[0]):
            modes, counts = np.unique(neigh_labels[i], return_counts=True)
            y_pred[i] = modes[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        check_is_fitted(self)
        X, y = validate_data(self, X=X, y=y, reset=False)
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """Monthly-based cross-validation splitter."""

    def __init__(self, time_col='index'):
        """Initialize the MonthlySplit cross-validator.

        Parameters
        ----------
        time_col : str, default='index'
            Column containing datetime information or 'index'.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations."""
        if self.time_col == 'index':
            dates = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column {self.time_col} not found in X")
            dates = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError(
                f"The column {self.time_col} is not a datetime."
            )

        dates = pd.Index(pd.to_datetime(dates))
        unique_months = dates.to_period('M').unique()
        return max(0, len(unique_months) - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        if self.time_col == 'index':
            dates = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column {self.time_col} not found in X")
            dates = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError(
                f"The column {self.time_col} is not a datetime."
            )

        dates = pd.Index(pd.to_datetime(dates))
        mont
