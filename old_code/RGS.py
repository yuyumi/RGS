from collections import Counter, defaultdict
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler


class RandomizedGreedySelection(BaseEstimator, RegressorMixin):
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
        RandomizedGreedySelection._validate_args(k, alpha, m, n_estimators)
        self.k = k
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self._validate_training_inputs(X, y)
        self.preprocessor = StandardScaler()
        X_scaled = self.preprocessor.fit_transform(X)
        _, self.p = X_scaled.shape
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
                correlations = np.abs(X_scaled[:,candidates].T @ residuals)
                best_feature = candidates[np.argmax(correlations)]
                selected.append(best_feature)
                X_subset = X_scaled[:, selected]
                lr = LinearRegression().fit(X_subset, y)
                residuals = y - lr.predict(X_subset)  # Update residual
            self.estimators.append(({"selected": selected,
                                     "lr" : lr}))

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.p
        X_scaled = self.preprocessor.transform(X)
        return np.mean([estimator["lr"].predict(X_scaled[:, estimator["selected"]]) for 
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


class FastRandomizedGreedySelection(BaseEstimator, RegressorMixin):
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

    tol : int, optional (default=0)
        The tolerance level for the minimum count of models to consider a feature set in the trajectory.
    
    resample : bool, optional (default=False)
        Whether to resample the counts of active sets at each iteration.
    
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Attributes:
    -----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the linear model.

    intercept_ : float
        Intercept of the linear model.

    trajectory : list of collections.Counter
        The trajectory of selected feature sets over iterations.    

    Methods:
    --------
    fit(X, y)
        Fit the RGS model to the training data.

    predict(X)
        Predict the target variable for the input data.

    step(active, X, y, m, n_iter, generator)
        Perform one step of greedy search starting with the feature set active and for n_iter repetitions.

    """

    def __init__(self, k, alpha=None, m=None, n_estimators=1000, tol=0, resample=False, random_state=None):
        FastRandomizedGreedySelection._validate_args(k, alpha, m, n_estimators)
        self.k = k
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.tol = tol
        self.resample = resample
        self.random_state = random_state

    def fit(self, X, y):
        self._validate_training_inputs(X, y)
        self.preprocessor = StandardScaler()
        X_scaled = self.preprocessor.fit_transform(X) # First center and normalize for fast compute
        _, self.p = X_scaled.shape
        generator = np.random.default_rng(self.random_state)
        self.trajectory = [Counter({frozenset({}) : self.n_estimators})]
        for j in range(self.k):
            active_sets = list(self.trajectory[j].keys()) # All feature subsets at step j
            counts = list(self.trajectory[j].values()) # How many times each feature subset appears in step j
            self.trajectory.append(Counter()) # Initialize the Counter for the feature subsets at step j+1
            if self.resample:
                total_count = sum(counts)
                proportions = np.array(counts) / total_count
                counts = generator.multinomial(self.n_estimators, proportions) # Resample the counts
            for active_set, count in zip(active_sets, counts):
                # For each feature set that occuring more than self.tol times, compute the next step
                if count > self.tol:
                    self.trajectory[j+1] += step(active_set, X_scaled, y, self.m, count, generator)
        # Average over the feature sets at the last step to compute the final estimator
        self.coef_ = np.zeros(self.p)
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        total_count = 0
        for selected, count in self.trajectory[self.k].items():
            selected = list(selected)
            beta, _, _, _ = lstsq(X_centered[:, selected], y)
            self.coef_[selected] += beta * count
            total_count += count
        self.coef_ = self.coef_ / total_count
        self.intercept_ = y.mean() - np.dot(X_mean, self.coef_)

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.p
        return X @ self.coef_ + self.intercept_
        
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


def step(active, X, y, m, n_iter, generator):

    active = list(active)
    p = X.shape[1]
    mask = np.ones(p, dtype=bool)
    mask[active] = False
    non_active = np.arange(p)[mask]

    # Step 1: Compute absolute correlations with current residual
    X_active = X[:, active]
    beta, _, _, _ = lstsq(X_active, y)
    residuals = y - X_active @ beta
    correlations = np.abs(X[:, non_active].T @ residuals)
    
    # Step 2: Create candidate feature subsets
    n_candidates = min(m, len(non_active))
    candidates = np.zeros((n_iter, n_candidates), dtype=int)
    for iter in range(n_iter):
        candidates[iter, :] = generator.choice(range(len(non_active)), size=n_candidates, replace=False)

    # Step 3: Compute the top candidate feature across each iteration
    candidate_correlations = correlations[candidates.flatten()]
    candidate_correlations = candidate_correlations.reshape(n_iter, n_candidates)
    max_index_in_subset = np.argmax(candidate_correlations, axis=1)
    selected_features = non_active[candidates[range(n_iter), max_index_in_subset]]

    # Step 4: Summarize the results in a Counter object
    selected_feature_counts = np.bincount(selected_features)
    selected_features_unique = np.nonzero(selected_feature_counts)[0]
    active = set(active)
    active_new_unique = [frozenset(active | {feat}) for feat in selected_features_unique]
    active_new_counts = Counter(dict(zip(active_new_unique, selected_feature_counts[selected_features_unique])))

    return active_new_counts