from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class FastRandomizedGreedySelectionCV(BaseEstimator, RegressorMixin):
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

    n_replications : int, optional (default=1000)
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

    def __init__(self, k_max, m_grid, n_replications=1000, n_resample_iter=0, random_state=None, cv=5):
        self.k_max = k_max
        self.m_grid = m_grid
        self.n_replications = n_replications
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state
        self.cv = cv
        self.cv_scores_ = None
        
    def _get_cv_splitter(self, X):
        """Convert cv parameter to a cv splitter object."""
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return self.cv

    def fit(self, X, y):
        # Initialize scores dictionary
        self.cv_scores_ = {k : {m : [] for m in self.m_grid} for k in range(1, self.k_max+1)}
        
        # Convert X and y to numpy arrays if they're pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Get CV splitter
        cv_splitter = self._get_cv_splitter(X)
        
        # Perform cross-validation
        for fold, (train_ids, val_ids) in enumerate(cv_splitter.split(X), 1):
            X_train = X[train_ids]
            X_val = X[val_ids]
            y_train = y[train_ids]
            y_val = y[val_ids]
            self.fit_fold(X_train, y_train, X_val, y_val)
        
        # Find best hyperparameters
        best_params = {}
        for k in range(1, self.k_max+1):
            mean_scores = {m: np.mean(self.cv_scores_[k][m]) for m in self.m_grid}
            best_m = min(mean_scores.items(), key=lambda x: x[1])
            best_params[k] = {'m': best_m[0], 'score': best_m[1]}
        
        # Find k with lowest overall MSE
        self.k_ = min(best_params.items(), key=lambda x: x[1]['score'])[0]
        self.m_ = best_params[self.k_]['m']
        
        # Fit final model with best parameters on full dataset
        self.final_fit(X, y, self.k_, self.m_)
        
        return self

    def final_fit(self, X, y, k, m):
        """Fit final model with best parameters on full dataset."""
        self.p = X.shape[1]
        generator = np.random.default_rng(self.random_state)
        
        # Center and normalize X for correlation computation
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        norms = np.sqrt(np.sum(X_centered ** 2, axis=0, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        X_scaled = X_centered / norms
        
        # Initialize coefficients
        self.coef_ = np.zeros(self.p)
        
        # Iteratively select features
        residuals = y.copy()
        selected_features = set()
        
        for _ in range(k):
            # Compute correlations with residuals
            correlations = np.abs(X_scaled.T @ residuals)
            
            # Select candidates
            n_candidates = min(m, self.p - len(selected_features))
            if n_candidates <= 0:
                break
                
            available_features = list(set(range(self.p)) - selected_features)
            candidates = generator.choice(available_features, size=n_candidates, replace=False)
            
            # Choose best feature
            best_feature = candidates[np.argmax(correlations[candidates])]
            selected_features.add(best_feature)
            
            # Update coefficients
            selected_list = list(selected_features)
            selected_X = X[:, selected_list]
            beta, _, _, _ = lstsq(selected_X, y)
            self.coef_[selected_list] = beta
            
            # Update residuals
            residuals = y - X @ self.coef_

    def predict(self, X):
        """Make predictions using the fitted model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X @ self.coef_
    
    def fit_fold(self, X_train, y_train, X_val, y_val):
        """Fit model on a single cross-validation fold."""
        self.p = X_train.shape[1]
        generator = np.random.default_rng(self.random_state)
        
        # Scale X for correlation computation
        norms = np.sqrt(np.sum(X_train ** 2, axis=0, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        X_scaled = X_train / norms
        
        # Initialize coefficients for each m value
        coef_ = {m: np.zeros((self.k_max, self.p)) for m in self.m_grid}
        
        # For each m value
        for m in self.m_grid:
            selected_features = set()
            residuals = y_train.copy()
            
            # Iteratively select features
            for k in range(self.k_max):
                # Compute correlations with residuals
                correlations = np.abs(X_scaled.T @ residuals)
                
                # Select candidates
                n_candidates = min(m, self.p - len(selected_features))
                if n_candidates <= 0:
                    break
                    
                available_features = list(set(range(self.p)) - selected_features)
                candidates = generator.choice(available_features, size=n_candidates, replace=False)
                
                # Choose best feature
                best_feature = candidates[np.argmax(correlations[candidates])]
                selected_features.add(best_feature)
                
                # Update coefficients
                selected_list = list(selected_features)
                selected_X = X_train[:, selected_list]
                beta, _, _, _ = lstsq(selected_X, y_train)
                coef_[m][k, selected_list] = beta
                
                # Compute validation score
                y_val_pred = X_val @ coef_[m][k]
                score = mean_squared_error(y_val, y_val_pred)
                self.cv_scores_[k+1][m].append(score)
                
                # Update residuals
                residuals = y_train - X_train @ coef_[m][k]
        
    @staticmethod
    def _validate_args(k, alpha, m, n_replications):
        assert isinstance(k, int) and k > 0
        assert isinstance(n_replications, int) and n_replications > 0
        assert alpha is None or m is None
        if alpha is not None:
            assert 0 < alpha <= 1
        else:
            assert isinstance(m, int) and m > 0

    def _validate_training_inputs(self, X, y):
        check_X_y(X, y)
        _, p = X.shape
        assert self.k_max <= p
        # Remove the m/alpha validation since we're using m_grid now
        assert all(m <= p for m in self.m_grid)  # Verify all m values are valid
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X, y

    def _get_new_feature_sets(self, M, M_comp, correlations, freqs_dict, generator):
        # Generate candidates for next step for each iteration
        feature_sets = {}
        for m, freq in freqs_dict.items():
            n_candidates = min(m, len(M_comp))
            candidates = np.zeros((freq, n_candidates), dtype=int)
            for iter in range(freq):
                candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
            # Compute the top candidate feature across each iteration
            candidate_correlations = correlations[candidates.flatten()].reshape(freq, n_candidates)
            max_index_in_subset = np.argmax(candidate_correlations, axis=1)
            psi_vals = M_comp[candidates[range(freq), max_index_in_subset]]
            # Summarize the results in a Counter object
            psi_freqs = np.bincount(psi_vals)
            psi_vals_unique = np.nonzero(psi_freqs)[0]
            M_new_unique = [frozenset(set(M) | {feat}) for feat in psi_vals_unique]
            M_new_freqs = Counter(dict(zip(M_new_unique, psi_freqs[psi_vals_unique])))
            feature_sets[m] = M_new_freqs
        return feature_sets
    
class RandomizedGreedySelection(BaseEstimator, RegressorMixin):
    """
    Randomized Greedy Selection (RGS) is a feature selection algorithm that selects a subset of features
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