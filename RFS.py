import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression


class RandomizedForwardSelection(BaseEstimator, RegressorMixin):
    """
    Randomized Forward Selection (RFS) is a feature selection algorithm that selects a subset of features
    based on their correlation with the target variable. RFS randomly selects a fixed number of candidate features
    at each iteration and chooses the feature with the highest correlation with the residuals. This process is
    repeated for a specified number of iterations to build an ensemble of linear regression models.

    Parameters:
    -----------
    k : int
        The number of features to select.

    alpha : float, optional (default=None)
        The fraction of features to randomly select as candidates at each iteration.
        Either `alpha` or `m` should be provided.

    m : int, optional (default=None)
        The fixed number of features to randomly select as candidates at each iteration.
        Either `alpha` or `m` should be provided.

    n_estimators : int, optional (default=1000)
        The number of linear regression models to build.

    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Attributes:
    -----------
    estimators : list
        A list of dictionaries, where each dictionary contains the selected features and the corresponding
        linear regression model.

    Methods:
    --------
    fit(X, y)
        Fit the RFS model to the training data.

    predict(X)
        Predict the target variable for the input data.

    """

    def __init__(self, k, alpha=None, m=None, n_estimators=1000, random_state=None):
        RandomizedForwardSelection._validate_args(k, alpha, m, n_estimators)
        self.k = k
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self._validate_training_inputs(X, y)
        _, self.p = X.shape
        generator = np.random.default_rng(self.random_state)
        self.estimators = []
        for _ in range(self.n_estimators):
            selected = []
            residuals = y.copy()
            for _ in range(self.k):
                # Randomly choose m candidate features
                candidates = generator.choice([i for i in range(self.p) if i not in selected], 
                                              size=min(self.m, self.p - len(selected)), replace=False)
                # Calculate correlations between candidates and residual
                correlations = np.abs([np.corrcoef(X[:, j], residuals)[0, 1] for j in candidates])
                best_feature = candidates[np.argmax(correlations)]
                selected.append(best_feature)
                X_subset = X[:, selected]
                lr = LinearRegression().fit(X_subset, y)
                residuals = y - lr.predict(X_subset)  # Update residual
            self.estimators.append(({"selected": selected,
                                     "lr" : lr}))

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.p
        return np.mean([estimator["lr"].predict(X[:, estimator["selected"]]) for 
                        estimator in self.estimators], axis=0)
        
    @staticmethod
    def _validate_args(k, alpha, m, n_estimators):
        assert isinstance(k, int) and k > 0
        assert isinstance(n_estimators, int) and n_estimators > 0
        assert alpha is None or m is None
        if alpha is not None:
            assert 0 < alpha <= 1
        else:
            assert isinstance(m, int) and m > 0

    def _validate_training_inputs(self, X, y):
        check_X_y(X, y)
        _, p = X.shape
        assert self.k <= p
        if self.m is None:
            self.m = np.ceil(self.alpha * p)
        assert self.m <= p