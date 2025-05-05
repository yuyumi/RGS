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
    
    # Extend Q by appending new column (more efficient than full reconstruction)
    Q_new = np.concatenate((Q, q_k_plus_1.reshape(-1, 1)), axis=1)
    
    # Extend R incrementally
    # First create new R with extended shape
    R_new = np.zeros((k+1, k+1))
    # Copy existing upper triangular part
    R_new[:k, :k] = R
    # Add new column
    R_new[:k, k] = r_k
    # Add diagonal element
    R_new[k, k] = q_k_plus_1_norm
    
    return np.ascontiguousarray(Q_new), np.ascontiguousarray(R_new)

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
            Ms = list(self.feature_sets[k].keys())   
            freqs = np.array(list(self.feature_sets[k].values()))

            if len(freqs) == 0 or freqs.sum() == 0:
                continue  # Skip this k value
            
            # Resample to speed up computation
            for _ in range(self.n_resample_iter):
                proportions = freqs / self.n_estimators
                freqs = generator.multinomial(self.n_estimators, proportions)
            valid_indices = np.where(freqs > 0)[0]
            Ms = [Ms[i] for i in valid_indices]
            freqs = freqs[freqs > 0]

            # Compute QR once per unique subset
            unique_Ms = list(set(Ms))
            qr_results = {}
            
            for unique_M in unique_Ms:
                M = np.array(unique_M, dtype=int)
                
                # Your existing QR computation logic:
                if M.size > 0:
                    x_first = np.ascontiguousarray(X_centered[:, M[0]])
                    q_norm = np.linalg.norm(x_first)
                    Q = x_first.reshape(-1, 1) / q_norm
                    R = np.ascontiguousarray(np.array([[q_norm]]))
                    
                    for j in range(1, len(M)):
                        x_next = X_centered[:, M[j]]
                        Q, R = update_qr(Q, R, x_next)
                    
                    beta = np.linalg.solve(R, Q.T @ y_centered)
                    residuals = y_centered - Q @ (Q.T @ y_centered)
                else:
                    beta = np.array([])
                    residuals = y_centered.copy()
                    Q = np.empty((n, 0))
                    R = np.empty((0, 0))
                
                qr_results[unique_M] = (Q, R, beta, residuals)

            coef_ = np.zeros(self.p)

            # print(f"k={k}")
            # print(f"len(Ms)={len(Ms)}")
            # print(f"len(freqs)={len(freqs)}")
            # print(f"valid_indices={valid_indices}")
            # print(f"len(valid_indices)={len(valid_indices)}")
            
            for i in range(len(Ms)):
                # print(f"i={i}, len(Ms)={len(Ms)}")
                # if i < len(Ms):
                #     print(f"About to access Ms[{i}]")
                #     print(f"Ms[{i}]={Ms[i]}")
                # else:
                #     print(f"ERROR: i={i} >= len(Ms)={len(Ms)}")
                M = Ms[i]
                # print(f"qr_results keys: {list(qr_results.keys())}")
                # print(f"qr_results has empty tuple? {() in qr_results}")
                if M in qr_results:
                    Q, R, beta, residuals = qr_results[M]
                else:
                    print(f"ERROR: M={M} not found in qr_results")
                    # Debug what happened
                    print(f"unique_Ms: {unique_Ms}")
                    print(f"Ms: {Ms}")
                # Q, R, beta, residuals = qr_results[M]

                # # QR factorization with incremental updates
                # if M.size > 0:
                #     # Start with first feature
                #     x_first = np.ascontiguousarray(X_centered[:, M[0]])
                #     q_norm = np.linalg.norm(x_first)
                #     Q = np.ascontiguousarray(x_first.reshape(-1, 1) / q_norm)
                #     R = np.array([[q_norm]])
                    
                #     # Incrementally add remaining features
                #     for j in range(1, len(M)):
                #         x_next = np.ascontiguousarray(X_centered[:, M[j]])
                #         Q, R = update_qr(Q, R, x_next)
                    
                #     # Solve for coefficients
                #     beta = np.linalg.solve(R, Q.T @ y_centered)
                #     residuals = y_centered - Q @ (Q.T @ y_centered)
                # else:
                #     beta = np.array([])
                #     residuals = y_centered.copy()
                #     Q = np.empty((n, 0))
                #     R = np.empty((0, 0))
                
                # print(f"k={k}, i={i}")
                # print(f"M={M}, type={type(M)}")
                # print(f"beta.shape={beta.shape}")
                # print(f"freqs[i]={freqs[i]}")
                # print(f"coef_.shape={coef_.shape}")

                if len(M) > 0:  # Check if M is non-empty tuple
                    # print(f"M: {M}")
                    # print(f"M type: {type(M)}")
                    # print(f"M contents types: {[type(x) for x in M]}")
                    try:
                        M_as_array = np.array(M, dtype=int)
                        # print(f"M_as_array: {M_as_array}")
                        # print(f"M_as_array.dtype: {M_as_array.dtype}")
                    except Exception as e:
                        print(f"Error converting M to array: {e}")
                    # M_as_array = np.array(M, dtype=int)
                    coef_[M_as_array] += beta * freqs[i]
                else:
                    # Don't try to index with empty array
                    assert len(beta) == 0
                
                if k < self.k_max:
                    # Get candidate features
                    mask = np.ones(self.p, dtype=bool)
                    if len(M) > 0:  # Only set mask if M is non-empty
                        mask[M_as_array] = False
                    # mask[M] = False
                    M_comp = np.arange(self.p)[mask]
                    
                    # Numba optimized forward selection computation
                    fs_values = RGS._compute_fs_values(
                        X_centered, residuals, M_comp, Q, feature_norms)
                    
                    # Generate new feature subsets
                    if len(M_comp) > 0:
                        self.feature_sets[k+1].update(self._get_new_feature_sets(
                        M, M_comp, fs_values, freqs[i], generator))
            
            # Calculate model parameters
            self.coef_.append(coef_ / self.n_estimators)
            self.intercept_.append(y_mean - np.dot(X.mean(axis=0), coef_ / self.n_estimators))

    @staticmethod
    # @jit(nopython=True)
    def _compute_fs_values(X_centered, residuals, M_comp, Q, feature_norms):
        """
        Compute forward selection criterion values using vectorized operations.
        """
        # Get all candidate features at once
        X_candidates = X_centered[:, M_comp]
        
        # Compute correlations using matrix multiplication (vectorized)
        correlations = np.abs(residuals @ X_candidates)
        
        # Check if Q is empty (first feature selection)
        if Q.shape[1] == 0:
            # Simply normalize by feature norms
            fs_values = correlations / feature_norms[M_comp]
        else:
            # Pre-compute Q transpose
            Q_T = Q.T
            
            # Compute all projections at once using matrix multiplication
            proj_matrix = Q_T @ X_candidates
            
            # Compute orthogonal components for all candidates
            x_orth = X_candidates - Q @ proj_matrix
            
            # Compute norms of orthogonal components
            orth_norms = np.linalg.norm(x_orth, axis=0)
            
            # Handle numerical issues where vectors are linearly dependent
            valid_mask = orth_norms > 1e-10
            fs_values = np.full(len(M_comp), -np.inf)
            fs_values[valid_mask] = correlations[valid_mask] / orth_norms[valid_mask]
        
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
        if len(M_comp) == 0:
            return {}
        # Sample candidate features for each iteration
        n_candidates = min(self.m, len(M_comp))
        if n_candidates == 0:
            return {}
        candidates = np.zeros((n_iter, n_candidates), dtype=int)
        for iter in range(n_iter):
            candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
        
        # Select best feature from each candidate subset using forward selection
        candidate_values = fs_values[candidates.flatten()].reshape(n_iter, n_candidates)
        if candidate_values.size == 0:
            return {}
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