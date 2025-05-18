from collections import Counter
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression
from scipy import linalg
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
    Optimized Random Greedy Search (RGS) for feature selection.
    Heavily vectorized for performance with large feature counts.
    
    Parameters
    ----------
    k_max : int
        The maximum number of features to select.
    
    m : int or None, optional (default=None)
        The number of candidates to randomly sample at each step.
        Either m or alpha must be provided.
    
    alpha : float or None, optional (default=None)
        The fraction of features to randomly sample at each step.
        Either m or alpha must be provided.
    
    n_estimators : int, optional (default=1000)
        Number of replicates (B).
    
    n_resample_iter : int, optional (default=1)
        Number of bootstrap resampling iterations.
    
    random_state : int or None, optional (default=None)
        Random number generator seed.
    """
    def __init__(self, k_max, m=None, alpha=None, n_estimators=1000, 
                 n_resample_iter=1, random_state=None):
        self.k_max = k_max
        self.m = m
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state
        
    def _validate_inputs(self, X, y):
        """Validate input data and parameters."""
        n_samples, n_features = X.shape
        
        # Validate feature selection parameters
        assert self.k_max > 0, "k_max must be positive"
        assert self.k_max <= n_features, "k_max cannot exceed number of features"
        
        # Set m based on alpha if not provided
        if self.m is None and self.alpha is not None:
            self.m = max(1, int(np.ceil(self.alpha * n_features)))
        
        # Validate m
        assert self.m is not None, "Either m or alpha must be provided"
        assert 0 < self.m <= n_features, "m must be between 1 and n_features"
        
        # Validate n_estimators
        assert self.n_estimators > 0, "n_estimators must be positive"
        
        # Check for specialized path
        # if self.m == n_features and self.n_estimators == 1:
        #     print("Using specialized forward selection path (m=p, B=1)")
        
        # Determine if we can use bit encoding (p ≤ 64)
        self.use_bits = n_features <= 64
        if not self.use_bits and n_features > 100:
            warnings.warn("Large feature count (>100) detected. Performance may be impacted.")
        
        return X, y
    
    def _fast_qr_update(self, Q, R, x_new):
        """
        Highly optimized rank-one update for QR decomposition.
        Uses vectorized operations for performance.
        """
        n, k = Q.shape
        
        # Fast projection using BLAS-optimized dot product
        # This avoids creating intermediate matrices
        r_k = np.dot(Q.T, x_new)
        
        # Fast orthogonalization without temporary arrays
        q_k_plus_1 = x_new - np.dot(Q, r_k)
        q_k_plus_1_norm = np.linalg.norm(q_k_plus_1)
        
        # Handle numerical issues
        if q_k_plus_1_norm < 1e-10:
            return Q, R
        
        # Normalize in-place
        q_k_plus_1 /= q_k_plus_1_norm
        
        # Extend matrices efficiently - pre-allocate for performance
        Q_new = np.zeros((n, k+1))
        Q_new[:, :k] = Q
        Q_new[:, k] = q_k_plus_1
        
        # Optimized R update
        R_new = np.zeros((k+1, k+1))
        if k > 0:
            R_new[:k, :k] = R
            R_new[:k, k] = r_k
        R_new[k, k] = q_k_plus_1_norm
        
        return Q_new, R_new
    
    def _vectorized_bootstrap_resample(self, feature_freqs, bootstrap_iter=0):
        """
        Highly optimized bootstrap resampling using vectorized operations.
        """
        # Initialize random generator with varying seed
        if self.random_state is not None:
            # Ensure different RNG state for each iteration
            seed = self.random_state + 10000 * bootstrap_iter
            generator = np.random.RandomState(seed)
        else:
            generator = np.random.RandomState()
        
        # Dictionary-based implementation
        if not feature_freqs:
            return feature_freqs
            
        # Extract keys and frequencies for vectorized operations
        features = list(feature_freqs.keys())
        freqs = np.array(list(feature_freqs.values()), dtype=np.float64)
        total = np.sum(freqs)
        
        if total == 0:
            return {}
        
        # Fast proportion calculation
        proportions = freqs / total
        
        # Fast multinomial sampling
        new_freqs = generator.multinomial(self.n_estimators, proportions)
        
        # Efficient dictionary creation
        # Only include non-zero frequencies
        non_zero_idx = np.nonzero(new_freqs)[0]
        result = {features[i]: new_freqs[i] for i in non_zero_idx}
        
        return result
    
    def _batch_process_candidates(self, X_centered, residuals, Q, M, frequency, feature_norms):
        """
        Highly vectorized batch processing of candidate features.
        Processes all candidates at once for maximum performance.
        """
        # Create consistent random seed from feature set and base seed
        set_hash = hash(M) if M else 0
        if self.random_state is not None:
            seed = self.random_state + set_hash % 10000
            generator = np.random.RandomState(seed)
        else:
            generator = np.random.RandomState()
        
        # Fast mask creation for unselected features
        unselected_mask = np.ones(self.n_features, dtype=bool)
        if len(M) > 0:
            unselected_mask[list(M)] = False
        M_comp = np.nonzero(unselected_mask)[0]
        
        if len(M_comp) == 0:
            return {}
        
        # Determine number of candidates
        n_candidates = min(self.m, len(M_comp))
        if n_candidates == 0:
            return {}
        
        # Pre-allocate array for all samples to minimize memory allocations
        all_candidates = []
        candidate_sets = []
        
        # Vectorized generation of all random samples at once
        # This is faster than generating one at a time
        all_samples = np.zeros((frequency, n_candidates), dtype=np.int32)
        for i in range(frequency):
            all_samples[i] = generator.choice(len(M_comp), size=n_candidates, replace=False)
            candidates = M_comp[all_samples[i]]
            candidate_sets.append(candidates)
            all_candidates.extend(candidates)
        
        # Get unique candidates for batch processing
        unique_candidates = np.unique(all_candidates)
        
        # Fast batch computation of selection criteria
        X_candidates = X_centered[:, unique_candidates]
        
        # Fast correlation computation using BLAS-optimized dot product
        correlations = np.abs(residuals @ X_candidates)
        
        # Vectorized computation of selection criteria
        if Q.shape[1] == 0:
            # First feature selection - vectorized normalization
            fs_values = correlations / feature_norms[unique_candidates]
        else:
            # Fast orthogonal component computation
            # Compute projection matrix efficiently
            proj_matrix = Q.T @ X_candidates
            
            # Vectorized orthogonalization
            if Q.shape[1] < X_candidates.shape[1]:
                # More efficient when Q has fewer columns than candidates
                x_orth = X_candidates - Q @ proj_matrix
            else:
                # More efficient when Q has more columns
                x_orth = np.empty_like(X_candidates)
                for i in range(X_candidates.shape[1]):
                    x_orth[:, i] = X_candidates[:, i] - Q @ proj_matrix[:, i]
            
            # Fast norm computation
            orth_norms = np.linalg.norm(x_orth, axis=0)
            
            # Vectorized handling of numerical issues
            fs_values = np.full(len(unique_candidates), -np.inf)
            valid_mask = orth_norms > 1e-10
            fs_values[valid_mask] = correlations[valid_mask] / orth_norms[valid_mask]
        
        # Create lookup for fast access
        criteria_lookup = {feat: value for feat, value in zip(unique_candidates, fs_values)}
        
        # Fast processing of all candidate sets
        new_feature_sets = {}
        
        # Process each candidate set to select best feature
        for candidates in candidate_sets:
            # Vectorized lookup of selection values
            candidate_values = np.array([criteria_lookup[feat] for feat in candidates])
            
            # Fast selection of best feature
            best_idx = np.argmax(candidate_values)
            best_feature = candidates[best_idx]
            
            # Create new feature set
            new_M = tuple(sorted(list(M) + [best_feature]))
            
            # Fast frequency update
            if new_M in new_feature_sets:
                new_feature_sets[new_M] += 1
            else:
                new_feature_sets[new_M] = 1
        
        return new_feature_sets
    
    def _process_feature_sets_in_batches(self, X_centered, y_centered, active_sets, feature_norms, qr_cache):
        """
        Process feature sets in batches for improved performance.
        Uses vectorized operations wherever possible.
        """
        coef_k = np.zeros(self.n_features)
        new_sets = {}
        
        # Sort feature sets by size for better cache locality
        sorted_sets = sorted(active_sets.items(), key=lambda x: (len(x[0]), x[0]))
        
        # Process all feature sets
        for M, freq in sorted_sets:
            # Get or compute QR with frequency tracking
            if M in qr_cache:
                Q, R, cache_freq = qr_cache[M]
            else:
                # Compute QR from scratch
                if len(M) > 0:
                    M_array = np.array(M, dtype=int)
                    X_M = X_centered[:, M_array]
                    # Use scipy's faster QR implementation
                    Q, R = linalg.qr(X_M, mode='economic', check_finite=False)
                else:
                    Q = np.empty((self.n_samples, 0))
                    R = np.empty((0, 0))
                cache_freq = freq
                qr_cache[M] = (Q, R, cache_freq)
            
            # Compute coefficients
            if len(M) > 0:
                M_array = np.array(M, dtype=int)
                # Fast triangular solver from scipy
                beta = linalg.solve_triangular(R, Q.T @ y_centered, 
                                              lower=False, check_finite=False)
                
                # Vectorized coefficient update
                coef_k[M_array] += beta * freq
            
            # Compute residuals
            if len(M) > 0:
                # Fast residual computation using BLAS-optimized operations
                residuals = y_centered - Q @ (Q.T @ y_centered)
            else:
                residuals = y_centered.copy()
            
            # Update QR cache frequency and clean up if needed
            qr_cache[M] = (Q, R, cache_freq - freq)
            if qr_cache[M][2] <= 0:
                del qr_cache[M]
            
            # Process candidates in batch if not at max features
            batch_results = self._batch_process_candidates(
                X_centered, residuals, Q, M, freq, feature_norms)
            
            # Update new feature sets
            for new_M, new_freq in batch_results.items():
                if new_M in new_sets:
                    new_sets[new_M] += new_freq
                else:
                    new_sets[new_M] = new_freq
                
                # Pre-compute QR for next iteration with frequency tracking
                if new_M in qr_cache:
                    # Update frequency if already in cache
                    Q_new, R_new, exist_freq = qr_cache[new_M]
                    qr_cache[new_M] = (Q_new, R_new, exist_freq + new_freq)
                else:
                    # Find the new feature (the one not in M)
                    new_feature = next(f for f in new_M if f not in M)
                    
                    # Use optimized rank-one update
                    Q_new, R_new = self._fast_qr_update(
                        Q, R, X_centered[:, new_feature])
                    qr_cache[new_M] = (Q_new, R_new, new_freq)
        
        return coef_k, new_sets
    
    def _fit_specialized_forward_selection(self, X, y):
        """
        Specialized implementation for m=p, B=1 case (standard forward selection).
        Highly optimized with vectorized operations.
        """
        n_samples, n_features = X.shape
        
        # Center data
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean
        
        # Initialize storage
        self.coef_ = []
        self.intercept_ = []
        
        # Pre-compute feature norms using vectorized operations
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        feature_norms[feature_norms < 1e-10] = 1.0
        
        # Initial empty model
        self.coef_.append(np.zeros(n_features))
        self.intercept_.append(y_mean)
        
        # Initialize QR factorization
        Q = np.empty((n_samples, 0))
        R = np.empty((0, 0))
        
        # Initial residuals
        residuals = y_centered.copy()
        
        # Track selected features
        selected = []
        unselected_mask = np.ones(n_features, dtype=bool)
        
        # Feature selection loop - vectorized operations
        for k in range(1, self.k_max + 1):
            # Check if any features remain
            if not np.any(unselected_mask):
                # No features left, copy previous model
                self.coef_.append(self.coef_[-1].copy())
                self.intercept_.append(self.intercept_[-1])
                continue
            
            # Get unselected feature indices - fast vectorized operation
            remaining = np.nonzero(unselected_mask)[0]
            
            # Evaluate all remaining features at once
            X_candidates = X_centered[:, remaining]
            
            # Fast vectorized correlation computation
            correlations = np.abs(residuals @ X_candidates)
            
            # Fast, vectorized selection criteria computation
            if Q.shape[1] == 0:
                # First feature - vectorized normalization
                fs_values = correlations / feature_norms[remaining]
            else:
                # Fast orthogonalization
                proj_matrix = Q.T @ X_candidates
                x_orth = X_candidates - Q @ proj_matrix
                orth_norms = np.linalg.norm(x_orth, axis=0)
                
                # Vectorized handling of numerical issues
                fs_values = np.full(len(remaining), -np.inf)
                valid_mask = orth_norms > 1e-10
                fs_values[valid_mask] = correlations[valid_mask] / orth_norms[valid_mask]
            
            # Fast selection of best feature
            best_idx = np.argmax(fs_values)
            best_feature = remaining[best_idx]
            
            # Update tracking
            selected.append(best_feature)
            unselected_mask[best_feature] = False
            
            # Update QR factorization with optimized rank-one update
            x_new = X_centered[:, best_feature]
            Q, R = self._fast_qr_update(Q, R, x_new)
            
            # Fast coefficient computation using triangular solve
            beta = linalg.solve_triangular(R, Q.T @ y_centered, 
                                         lower=False, check_finite=False)
            
            # Update model - vectorized coefficient assignment
            coef_k = np.zeros(n_features)
            coef_k[selected] = beta
            self.coef_.append(coef_k)
            self.intercept_.append(y_mean)
            
            # Update residuals for next iteration
            residuals = y_centered - Q @ (Q.T @ y_centered)
        
        return self
    
    def fit(self, X, y):
        """
        Fit RGS with optimized implementation.
        Highly vectorized for performance.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns fitted estimator
        """
        # Validate inputs
        X, y = self._validate_inputs(X, y)
        self.n_samples, self.n_features = X.shape
        
        # Specialized path for m=p, B=1 case
        if self.m == self.n_features and self.n_estimators == 1:
            return self._fit_specialized_forward_selection(X, y)
        
        # Initialize data with vectorized operations
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean
        
        # Pre-compute feature norms using vectorized operations
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        feature_norms[feature_norms < 1e-10] = 1.0
        
        # Initialize feature tracking
        if self.use_bits:
            # Bit-encoded implementation
            feature_freqs = np.zeros(2**min(64, self.n_features), dtype=int)
            feature_freqs[0] = self.n_estimators  # Empty set
        else:
            # Dictionary-based implementation
            feature_sets = [{} for _ in range(self.k_max + 2)]  # +2 to avoid index errors
            feature_sets[0][tuple()] = self.n_estimators
        
        # Initialize model storage
        self.coef_ = []
        self.intercept_ = []
        
        # QR cache with frequency tracking
        # {feature_tuple: (Q, R, remaining_frequency)}
        qr_cache = {}
        qr_cache[tuple()] = (np.empty((self.n_samples, 0)), 
                             np.empty((0, 0)),
                             self.n_estimators)
        
        # Main loop over feature counts
        for k in range(self.k_max + 1):
            # print(f"Processing k={k}")
            
            # Get current feature sets
            if self.use_bits:
                # Extract sets of size k from bit array
                active_sets = {}
                for encoding in np.nonzero(feature_freqs)[0]:
                    features = self._decode_features(encoding)
                    if len(features) == k:
                        active_sets[features] = feature_freqs[encoding]
            else:
                active_sets = feature_sets[k].copy()
            
            # Apply bootstrap resampling (with fixed strength - faithful to paper)
            if k > 0 and self.n_resample_iter > 0:
                for iter_idx in range(self.n_resample_iter):
                    if self.use_bits:
                        feature_freqs = self._vectorized_bootstrap_resample(
                            feature_freqs, iter_idx)
                        
                        # Update active_sets from resampled frequencies
                        active_sets = {}
                        for encoding in np.nonzero(feature_freqs)[0]:
                            features = self._decode_features(encoding)
                            if len(features) == k:
                                active_sets[features] = feature_freqs[encoding]
                    else:
                        active_sets = self._vectorized_bootstrap_resample(
                            active_sets, iter_idx)
            
            # Process feature sets in batches
            coef_k, new_sets = self._process_feature_sets_in_batches(
                X_centered, y_centered, active_sets, feature_norms, qr_cache)
            
            # Update feature sets for next iteration - ONLY if not at max features
            if k < self.k_max:
                # Update feature sets for next iteration
                if self.use_bits:
                    # Clear frequencies for features of size k
                    for encoding in np.nonzero(feature_freqs)[0]:
                        if self._count_bits(encoding) == k:
                            feature_freqs[encoding] = 0
                    
                    # Update bit array with new frequencies
                    for M, freq in new_sets.items():
                        if freq > 0:
                            encoding = self._encode_features(M)
                            feature_freqs[encoding] = freq
                else:
                    # Update dictionary
                    feature_sets[k+1] = new_sets
            
            # Compute final coefficients
            self.coef_.append(coef_k / self.n_estimators)
            self.intercept_.append(y_mean - np.dot(X_mean, coef_k / self.n_estimators))
            
            # Memory optimization: clear unnecessary data
            if k >= 2:
                if not self.use_bits:
                    # Clear old feature sets to save memory
                    feature_sets[k-2] = {}
            
            # Progress report
            # print(f"Completed k={k}, active sets: {len(active_sets)}, new sets: {len(new_sets)}")
        
        return self
    
    def predict(self, X, k=None):
        """
        Predict using the RGS model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
        k : int, optional (default=None)
            Number of features to use. If None, uses k_max.
        
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values
        """
        if k is None:
            k = self.k_max
        else:
            k = min(k, self.k_max)
        
        return X @ self.coef_[k] + self.intercept_[k]
    
    # Bit-encoding support methods
    def _encode_features(self, features):
        """Encode feature set as a bit vector."""
        encoding = 0
        for f in features:
            encoding |= (1 << f)
        return encoding
    
    def _decode_features(self, encoding):
        """Decode bit vector to feature set."""
        return tuple(i for i in range(self.n_features) 
                    if encoding & (1 << i))
    
    def _count_bits(self, encoding):
        """Count number of set bits (features)."""
        return bin(encoding).count('1')
        
    # @staticmethod
    # def _validate_args(k, alpha, m, n_estimators):
    #     assert isinstance(k, int) and k > 0
    #     assert isinstance(n_estimators, int) and n_estimators > 0
    #     assert alpha is None or m is None
    #     if alpha is not None:
    #         assert 0 < alpha <= 1
    #     else:
    #         assert isinstance(m, int) and m > 0

    # def _validate_training_inputs(self, X, y):
    #     check_X_y(X, y)
    #     _, p = X.shape
    #     assert self.k_max <= p
    #     if self.m is None:
    #         self.m = np.ceil(self.alpha * p)
    #     assert self.m <= p
    #     if isinstance(X, pd.DataFrame):
    #         X = X.values
    #     return X, y

    # def _get_new_feature_sets(self, M, M_comp, fs_values, n_iter, generator):
    #     """
    #     Generate new feature sets based on forward selection criterion.
        
    #     Parameters:
    #     -----------
    #     M : list
    #         Current feature set
    #     M_comp : array
    #         Indices of features not in M
    #     fs_values : array
    #         Forward selection criterion values for each feature in M_comp
    #     n_iter : int
    #         Number of iterations to perform
    #     generator : Generator
    #         Random number generator
            
    #     Returns:
    #     --------
    #     Counter
    #         New feature sets with their frequencies
    #     """
    #     if len(M_comp) == 0:
    #         return {}
    #     # Sample candidate features for each iteration
    #     n_candidates = min(self.m, len(M_comp))
    #     if n_candidates == 0:
    #         return {}
    #     candidates = np.zeros((n_iter, n_candidates), dtype=int)
    #     for iter in range(n_iter):
    #         candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
        
    #     # Select best feature from each candidate subset using forward selection
    #     candidate_values = fs_values[candidates.flatten()].reshape(n_iter, n_candidates)
    #     if candidate_values.size == 0:
    #         return {}
    #     max_index_in_subset = np.argmax(candidate_values, axis=1)
    #     psi_vals = M_comp[candidates[range(n_iter), max_index_in_subset]]
        
    #     # Count frequencies of selected features
    #     psi_freqs = np.bincount(psi_vals)
    #     psi_vals_unique = np.nonzero(psi_freqs)[0]
        
    #     # Create new feature sets by adding selected features to M
    #     M_new_unique = [tuple(sorted(np.append(M, feat))) for feat in psi_vals_unique]
    #     M_new_freqs = dict(zip(M_new_unique, psi_freqs[psi_vals_unique]))
        
    #     return M_new_freqs

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