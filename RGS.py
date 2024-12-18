from collections import Counter, defaultdict
import numpy as np
import pandas as pd

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
    FastRandomizedGreedySelection is a feature selection algorithm that selects a subset of features
    based on their correlation with the target variable. RGS randomly selects a fixed number of candidate features
    at each iteration and chooses the feature with the highest correlation with the residuals. This process is
    repeated for a specified number of iterations to build an ensemble of linear regression models.

    Parameters
    ----------
    k_max : int
        The maximum number of features to select.

    alpha : float, optional (default=None)
        The fraction of features to randomly select as candidates at each iteration.
        Either `alpha` or `m` should be provided.

    m : int, optional (default=None)
        The fixed number of features to randomly select as candidates at each iteration.
        Either `alpha` or `m` should be provided.

    n_estimators : int, optional (default=1000)
        The number of linear regression models to build.

    n_resample_iter : int, default=0
        The number of resampling iterations to perform to speed up computation.

    random_state : int or None, default=None
        The seed of the random number generator.

    Attributes
    ----------
    coef_ : list of ndarray
        The coefficients of the linear model for each step k.

    intercept_ : list of float
        The intercept of the linear model for each step k.

    feature_sets : list of Counter
        The feature subsets selected at each step k.

    Methods
    -------
    fit(X, y)
        Fit the model to the data.

    predict(X, k=None)
        Predict using the linear model for step k.
    """


    def __init__(self, k_max, alpha=None, m=None, n_estimators=1000, n_resample_iter=0, random_state=None):
        FastRandomizedGreedySelection._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Initialize
        X, y = self._validate_training_inputs(X, y)
        X_centered = X - X.mean(axis=0)
        X_scaled = X_centered / np.sqrt(np.sum((X ** 2), axis=0))
        _, self.p = X_scaled.shape
        generator = np.random.default_rng(self.random_state)
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] += Counter({frozenset({}) : self.n_estimators})
        self.coef_ = []
        self.intercept_ = []
        for k in range(self.k_max + 1):
            Ms = np.array(list(self.feature_sets[k].keys())) # All feature subsets appearing at step k
            freqs = np.array(list(self.feature_sets[k].values())) # How many times each feature subset appears in step k
            # Resample to speed up computation
            for _ in range(self.n_resample_iter):
                proportions = freqs / self.n_estimators
                freqs = generator.multinomial(self.n_estimators, proportions)
            Ms = Ms[freqs > 0]
            freqs = freqs[freqs > 0]
            coef_ = np.zeros(self.p)
            for i in range(len(Ms)):
                M = list(Ms[i])
                # Compute least squares solution of y on X_M
                beta, _, _, _ = lstsq(X_centered[:, M], y)
                residuals = y - X_centered[:, M] @ beta
                coef_[M] += beta * freqs[i]
                if k < self.k_max:
                    # Compute residual correlation with features not in M
                    mask = np.ones(self.p, dtype=bool)
                    mask[M] = False
                    M_comp = np.arange(self.p)[mask]
                    correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                    # Generate new feature subsets
                    self.feature_sets[k+1] += self._get_new_feature_sets(M, M_comp, correlations, freqs[i], generator)
            # Calculate parameters
            self.coef_.append(coef_ / self.n_estimators)
            self.intercept_.append(y.mean() - np.dot(X.mean(axis=0), coef_ / self.n_estimators))

    def predict(self, X, k=None):
        assert X.ndim == 2 and X.shape[1] == self.p
        if k is None:
            k = self.k_max
        return X @ self.coef_[k] + self.intercept_[k]
        
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
        assert self.k_max <= p
        if self.m is None:
            self.m = np.ceil(self.alpha * p)
        assert self.m <= p
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X, y

    def _get_new_feature_sets(self, M, M_comp, correlations, n_iter, generator):
        # Generate candidates for next step for each iteration
        n_candidates = min(self.m, len(M_comp))
        candidates = np.zeros((n_iter, n_candidates), dtype=int)
        for iter in range(n_iter):
            candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
        # Compute the top candidate feature across each iteration
        candidate_correlations = correlations[candidates.flatten()].reshape(n_iter, n_candidates)
        max_index_in_subset = np.argmax(candidate_correlations, axis=1)
        psi_vals = M_comp[candidates[range(n_iter), max_index_in_subset]]
        # Summarize the results in a Counter object
        psi_freqs = np.bincount(psi_vals)
        psi_vals_unique = np.nonzero(psi_freqs)[0]
        M_new_unique = [frozenset(set(M) | {feat}) for feat in psi_vals_unique]
        M_new_freqs = Counter(dict(zip(M_new_unique, psi_freqs[psi_vals_unique])))
        return M_new_freqs
    
    # def fit_stability(self):
    #     self.stability_scores = np.zeros(self.k_max + 1)
    #     for k in range(1, self.k_max + 1):
    #         score = 0
    #         for M1, freq1 in self.feature_sets[k].items():
    #             for M2, freq2 in self.feature_sets[k].items():
    #                 score += (k - len(M1 & M2)) * freq1 * freq2
    #         score = score / (self.n_estimators ** 2)
    #         self.stability_scores[k] = score
    #     return self.stability_scores