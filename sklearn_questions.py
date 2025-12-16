"""Assignment - making a sklearn estimator and cv splitter.
The goal of this assignment is to implement by yourself:
- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.
Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.
the training sample X_k which is the closest to X_i. We measure proximity
with the Euclidean distance. The model will be evaluated with the accuracy
(average number of samples corectly classified). You need to implement the
`fit`, `predict` and `score` methods for this class. The code you write
should pass the test we implemented. You can run the tests by calling at
the root of the repo `pytest test_sklearn_questions.py`. Note that to be
fully valid, a scikit-learn estimator needs to check that the input given
to `fit` and `predict` are correct using the `validate_data,
check_is_fitted` functions imported in this file.
You can find more information on how they should be used in the following
doc:
https://scikit-learn.org/stable/developers/develop.html
#rolling-your-own-estimator.
Make sure to use them to pass
`test_nearest_neighbor_check_estimator`.
Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.
We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.
The data to split should contain the index or one column in datatime
format. Then the aim is to split the data between train and test sets when
for each pair of successive months, we learn on the first and predict of
the following. For example if you have data distributed from november
2020 to march 2021, you have have 4 splits. The first split will allow to
learn on november data and predict on december data, the second split to
learn december and predict on january etc.
We also ask you to respect the pep8 convention: https://pep8.org. This
will be enforced with `flake8`. You can check that there is no flake8
errors by calling `flake8` at the root of the repo.
Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.
Hints
-----
- You can use the function:
from sklearn.metrics.pairwise import pairwise_distances
to compute distances between 2 sets of samples.
"""
"""Custom scikit-learn utilities.
This module provides a custom K-Nearest Neighbors classifier and a
time-based cross-validation splitter.
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
        months = dates.to_period('M')
        unique_months = sorted(months.unique())

        n_splits = self.get_n_splits(X, y, groups)
        if n_splits == 0:
            return

        for i in range(n_splits):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = months == train_month
            test_mask = months == test_month

            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]

            yield idx_train, idx_test
