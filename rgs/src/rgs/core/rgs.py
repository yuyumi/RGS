from collections import Counter
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer

import numba as nb
from numba import jit, float64, int64, boolean

@jit(nopython=True)
def orthogonalize_vector(x, Q):
    """Orthogonalize vector x against columns of Q using Numba for speed"""
    x_orth = x.copy()
    
    for i in range(Q.shape[1]):
        proj = np.dot(Q[:, i], x)
        x_orth = x_orth - proj * Q[:, i]
        
    return x_orth


# Numba-optimized function for incremental QR update
@jit(nopython=True)
def update_qr(Q, R, x_new):
    n, k = Q.shape
    
    # Project x_new onto current Q
    r_k = Q.T @ x_new
    
    # Compute orthogonal component
    q_k_plus_1 = x_new - Q @ r_k
    q_k_plus_1_norm = np.linalg.norm(q_k_plus_1)
    
    # Handle numerical issues
    if q_k_plus_1_norm < 1e-10:
        # New vector is linearly dependent, return original Q and R
        return Q, R
    
    # Normalize the new orthogonal component
    q_k_plus_1 = q_k_plus_1 / q_k_plus_1_norm
    
    # Construct new Q and R matrices
    Q_new = np.empty((n, k+1))
    Q_new[:, :k] = Q
    Q_new[:, k] = q_k_plus_1
    
    R_new = np.zeros((k+1, k+1))
    R_new[:k, :k] = R
    R_new[:k, k] = r_k
    R_new[k, k] = q_k_plus_1_norm
    
    Q_new = np.ascontiguousarray(Q_new)
    R_new = np.ascontiguousarray(R_new)
    return Q_new, R_new

class RGS(BaseEstimator, RegressorMixin):
    """
    RGS is a feature selection algorithm that selects a subset of features
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
        RGS._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Initialize
        X, y = self._validate_training_inputs(X, y)
        X_centered = np.ascontiguousarray(X - X.mean(axis=0))
        y_centered = np.ascontiguousarray(y - y.mean())
        y_mean = y.mean()
        
        n, self.p = X_centered.shape
        generator = np.random.default_rng(self.random_state)
        self.feature_sets = [{} for _ in range(self.k_max + 1)]
        self.feature_sets[0][tuple()] = self.n_estimators
        self.coef_ = []
        self.intercept_ = []
        
        # Pre-compute feature norms for normalization
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        # Avoid division by zero
        feature_norms[feature_norms < 1e-10] = 1.0
        
        for k in range(self.k_max + 1):
            Ms = np.array(list(self.feature_sets[k].keys()), dtype=object)  
            freqs = np.array(list(self.feature_sets[k].values()))
            
            # Resample to speed up computation
            for _ in range(self.n_resample_iter):
                proportions = freqs / self.n_estimators
                freqs = generator.multinomial(self.n_estimators, proportions)
            Ms = Ms[freqs > 0]
            freqs = freqs[freqs > 0]
            coef_ = np.zeros(self.p)
            
            for i in range(len(Ms)):
                M = np.array(Ms[i], dtype=int)


                # QR factorization with incremental updates
                if M.size > 0:
                    # Start with first feature
                    x_first = X_centered[:, M[0]]
                    q_norm = np.linalg.norm(x_first)
                    Q = np.ascontiguousarray(x_first.reshape(-1, 1) / q_norm)
                    R = np.array([[q_norm]])
                    
                    # Incrementally add remaining features
                    for j in range(1, len(M)):
                        x_next = np.ascontiguousarray(X_centered[:, M[j]])
                        Q, R = update_qr(Q, R, x_next)
                    
                    # Solve for coefficients
                    beta = np.linalg.solve(R, Q.T @ y_centered)
                    residuals = y_centered - Q @ (Q.T @ y_centered)
                else:
                    beta = np.array([])
                    residuals = y_centered.copy()
                    Q = np.empty((n, 0))
                    R = np.empty((0, 0))
                
                # print(f"k={k}, i={i}")
                # print(f"M={M}, type={type(M)}")
                # print(f"beta.shape={beta.shape}")
                # print(f"freqs[i]={freqs[i]}")
                # print(f"coef_.shape={coef_.shape}")

                coef_[M] += beta * freqs[i]
                
                if k < self.k_max:
                    # Get candidate features
                    mask = np.ones(self.p, dtype=bool)
                    mask[M] = False
                    M_comp = np.arange(self.p)[mask]
                    
                    # Numba optimized forward selection computation
                    fs_values = RGS._compute_fs_values(
                        X_centered, residuals, M_comp, Q, feature_norms)
                    
                    # Generate new feature subsets
                    self.feature_sets[k+1].update(self._get_new_feature_sets(
                        M, M_comp, fs_values, freqs[i], generator))
            
            # Calculate model parameters
            self.coef_.append(coef_ / self.n_estimators)
            self.intercept_.append(y_mean - np.dot(X.mean(axis=0), coef_ / self.n_estimators))

    @staticmethod
    @jit(nopython=True)
    def _compute_fs_values(X_centered, residuals, M_comp, Q, feature_norms):
        """
        Compute forward selection criterion values using optimized Numba
        """
        fs_values = np.zeros(len(M_comp))
        n_candidates = len(M_comp)
        
        # Precompute all correlations with residuals at once
        X_candidates = np.ascontiguousarray(X_centered[:, M_comp])
        correlations = np.zeros(n_candidates)
        for idx in range(n_candidates):
            correlations[idx] = np.abs(np.dot(residuals, X_candidates[:, idx]))
        
        # Check if Q is empty (first feature selection)
        if Q.shape[1] == 0:
            # If Q is empty, just return correlations normalized by feature norms
            for idx in nb.prange(n_candidates):
                fs_values[idx] = correlations[idx] / feature_norms[M_comp[idx]]
            return fs_values
        
        # For non-empty Q, compute orthogonal components
        # Precompute Q'X for all candidate features
        QTX = np.zeros((Q.shape[1], n_candidates))
        for q_idx in range(Q.shape[1]):
            for x_idx in range(n_candidates):
                QTX[q_idx, x_idx] = np.dot(Q[:, q_idx], X_candidates[:, x_idx])
        
        # Compute orthogonal components and their norms in parallel
        for idx in nb.prange(n_candidates):
            x_j = X_candidates[:, idx]
            
            # Compute orthogonal component using precomputed Q'X
            x_j_orth = x_j.copy()
            for q_idx in range(Q.shape[1]):
                proj = QTX[q_idx, idx]
                x_j_orth = x_j_orth - proj * Q[:, q_idx]
            
            # Compute norm of orthogonal component
            orth_norm = np.linalg.norm(x_j_orth)
            
            # Skip if linearly dependent
            if orth_norm < 1e-10:
                fs_values[idx] = -np.inf
            else:
                # Forward selection criterion
                fs_values[idx] = correlations[idx] / orth_norm
        
        return fs_values

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

    def _get_new_feature_sets(self, M, M_comp, fs_values, n_iter, generator):
        """
        Generate new feature sets based on forward selection criterion.
        
        Parameters:
        -----------
        M : list
            Current feature set
        M_comp : array
            Indices of features not in M
        fs_values : array
            Forward selection criterion values for each feature in M_comp
        n_iter : int
            Number of iterations to perform
        generator : Generator
            Random number generator
            
        Returns:
        --------
        Counter
            New feature sets with their frequencies
        """
        # Sample candidate features for each iteration
        n_candidates = min(self.m, len(M_comp))
        candidates = np.zeros((n_iter, n_candidates), dtype=int)
        for iter in range(n_iter):
            candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
        
        # Select best feature from each candidate subset using forward selection
        candidate_values = fs_values[candidates.flatten()].reshape(n_iter, n_candidates)
        max_index_in_subset = np.argmax(candidate_values, axis=1)
        psi_vals = M_comp[candidates[range(n_iter), max_index_in_subset]]
        
        # Count frequencies of selected features
        psi_freqs = np.bincount(psi_vals)
        psi_vals_unique = np.nonzero(psi_freqs)[0]
        
        # Create new feature sets by adding selected features to M
        M_new_unique = [tuple(sorted(np.append(M, feat))) for feat in psi_vals_unique]
        M_new_freqs = dict(zip(M_new_unique, psi_freqs[psi_vals_unique]))
        
        return M_new_freqs

class RGSCV(BaseEstimator, RegressorMixin):
    """
    Cross-validation wrapper for RGS with custom scoring support.
    
    Parameters
    ----------
    k_max : int
        The maximum number of features to select.
    
    m_grid : list of int
        Grid of m values to try, where m is the number of candidates
        to randomly select at each iteration.
    
    n_estimators : int, default=1000
        Number of estimators for the ensemble.
        
    n_resample_iter : int, default=0
        Number of resampling iterations.
        
    random_state : int or None, default=None
        Random number generator seed.
        
    cv : int, default=5
        Number of cross-validation folds.
        
    scoring : string, callable, or None, default=None
        Scoring method to use. If None, defaults to 'neg_mean_squared_error'.
        If string, uses sklearn's scoring methods.
        If callable, expects a function with signature scorer(y_true, y_pred).
    """
    def __init__(self, k_max, m_grid, n_estimators=1000, n_resample_iter=0, 
                 k_grid=None, cv=5, scoring=None, random_state=None):
        self.k_max = k_max
        self.m_grid = m_grid
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.k_grid = range(1, k_max+1) if k_grid is None else k_grid
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring  # This will be a scorer factory function
        
    def _get_scorer(self, k):
        """Get a scoring function for the current k value."""
        if self.scoring is None:
            return get_scorer('neg_mean_squared_error')
        elif isinstance(self.scoring, str):
            return get_scorer(self.scoring)
        elif callable(self.scoring):
            # self.scoring is our make_k_scorer function
            scorer = self.scoring(k)
            return scorer
        else:
            raise ValueError("scoring should be None, a string, or a callable")
        
    def fit(self, X, y):
        """Fit the model using cross-validation to select the best m."""
        # Convert inputs if needed
        # if isinstance(X, pd.DataFrame):
        #     X = X.values
        # if isinstance(y, pd.Series):
        #     y = y.values

        # Initialize scores dictionary
        self.cv_scores_ = {k: {m: [] for m in self.m_grid} 
                        for k in range(1, self.k_max + 1)}
        
        if self.cv == 1:
            # No CV - use full dataset directly
            for m in self.m_grid:
                model = RGS(
                    k_max=self.k_max,
                    m=m,
                    n_estimators=self.n_estimators,
                    n_resample_iter=self.n_resample_iter,
                    random_state=self.random_state
                )
                model.fit(X, y)
                
                # Evaluate for each k
                for k in self.k_grid:
                        y_pred = model.predict(X, k=k)
                        # Get scorer for current k
                        scorer = self._get_scorer(k)
                        score = scorer._score_func(y, y_pred)
                        self.cv_scores_[k][m] = [score]           
        else:
                
            # Setup CV splitter
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, 
                            random_state=self.random_state) if isinstance(self.cv, int) else self.cv
            
            # Perform CV
            for train_idx, val_idx in cv_splitter.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                for m in self.m_grid:
                    model = RGS(
                        k_max=self.k_max,
                        m=m,
                        n_estimators=self.n_estimators,
                        n_resample_iter=self.n_resample_iter,
                        random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    
                    # Evaluate for each k
                    for k in self.k_grid:
                        y_pred = model.predict(X_val, k=k)
                        # Get scorer for current k
                        scorer = self._get_scorer(k)
                        score = scorer._score_func(y_val, y_pred)
                        self.cv_scores_[k][m].append(score)
    
        # Find best parameters
        best_params = {}
        for k in self.k_grid:
            mean_scores = {m: np.mean(self.cv_scores_[k][m]) for m in self.m_grid}
            best_m = max(mean_scores.items(), key=lambda x: x[1])[0]
            best_params[k] = {'m': best_m, 'score': mean_scores[best_m]}
        
        # Find optimal k
        self.k_ = max(best_params.items(), key=lambda x: x[1]['score'])[0]
        self.m_ = best_params[self.k_]['m']
        
        # Fit final model with full k_max
        self.model_ = RGS(
            k_max=self.k_max,
            m=self.m_,
            n_estimators=self.n_estimators,
            n_resample_iter=self.n_resample_iter,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        return self
    
    def predict(self, X, k=None):
        """Make predictions using the fitted model with best parameters."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Use best k if k is not specified, otherwise use specified k
        k_to_use = self.k_ if k is None else k
        return self.model_.predict(X, k=k_to_use)