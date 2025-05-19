from collections import Counter
import warnings
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
        
        # First pass: Compute QR, coefficients, and residuals for all feature sets
        all_data = []  # Will store (M, freq, Q, R, residuals, unselected_features)
        
        for M, freq in sorted_sets:
            # Get or compute QR with frequency tracking
            if M in qr_cache:
                Q, R, cache_freq = qr_cache[M]
            else:
                # Compute QR from scratch
                if len(M) > 0:
                    M_array = np.array(M, dtype=int)
                    X_M = X_centered[:, M_array]
                    Q, R = linalg.qr(X_M, mode='economic', check_finite=False)
                else:
                    Q = np.empty((self.n_samples, 0))
                    R = np.empty((0, 0))
                cache_freq = freq
                qr_cache[M] = (Q, R, cache_freq)
            
            # Compute coefficients
            if len(M) > 0:
                M_array = np.array(M, dtype=int)
                beta = linalg.solve_triangular(R, Q.T @ y_centered, lower=False, check_finite=False)
                coef_k[M_array] += beta * freq
            
            # Compute residuals
            residuals = y_centered - Q @ (Q.T @ y_centered) if len(M) > 0 else y_centered.copy()
            
            # Update QR cache frequency
            qr_cache[M] = (Q, R, cache_freq - freq)
            if qr_cache[M][2] <= 0:
                del qr_cache[M]
            
            # Get unselected features
            unselected_mask = np.ones(self.n_features, dtype=bool)
            if len(M) > 0:
                unselected_mask[list(M)] = False
            M_comp = np.nonzero(unselected_mask)[0]
            
            # Store all data needed for processing candidates
            all_data.append((M, freq, Q, residuals, M_comp))
        
        # Second pass: Process all candidate features in a single batch for each feature set
        # This replaces the million+ individual calls to _batch_process_candidates
        for M, freq, Q, residuals, M_comp in all_data:
            if len(M_comp) == 0:
                continue
            
            # Create consistent random seed from feature set and base seed
            set_hash = hash(M) if M else 0
            if self.random_state is not None:
                seed = self.random_state + set_hash % 10000
                generator = np.random.RandomState(seed)
            else:
                generator = np.random.RandomState()
            
            # Determine number of candidates
            n_candidates = min(self.m, len(M_comp))
            if n_candidates == 0:
                continue
            
            # Generate random samples all at once (instead of one at a time)
            all_candidates = set()
            all_samples = []
            
            for i in range(freq):
                sample_indices = generator.choice(len(M_comp), size=n_candidates, replace=False)
                sample = M_comp[sample_indices]
                all_samples.append(sample)
                all_candidates.update(sample)
            
            # Convert to array for vectorized operations
            all_candidates = np.array(list(all_candidates))
            
            # Get candidate features
            X_candidates = X_centered[:, all_candidates]
            
            # Fast correlation computation
            correlations = np.abs(residuals @ X_candidates)
            
            # Vectorized computation of selection criteria
            if Q.shape[1] == 0:
                # First feature - simple correlation/norm
                fs_values = correlations / feature_norms[all_candidates]
            else:
                # Orthogonalization for subsequent features
                proj_matrix = Q.T @ X_candidates
                x_orth = X_candidates - Q @ proj_matrix
                orth_norms = np.linalg.norm(x_orth, axis=0)
                
                # Handle numerical issues
                fs_values = np.full(len(all_candidates), -np.inf)
                valid_mask = orth_norms > 1e-10
                fs_values[valid_mask] = correlations[valid_mask] / orth_norms[valid_mask]
            
            # Map features to their selection values for quick lookup
            criteria_lookup = {feat: val for feat, val in zip(all_candidates, fs_values)}
            
            # Process all samples for this feature set at once
            for sample in all_samples:
                # Get selection values for this sample
                sample_values = np.array([criteria_lookup[feat] for feat in sample])
                
                # Select best feature
                best_idx = np.argmax(sample_values)
                best_feature = sample[best_idx]
                
                # Create new feature set
                new_M = tuple(sorted(list(M) + [best_feature]))
                
                # Update new feature sets
                if new_M in new_sets:
                    new_sets[new_M] += 1
                else:
                    new_sets[new_M] = 1
                    
                # Pre-compute QR for next iteration (optional optimization)
                if new_M in qr_cache:
                    # Update frequency if already in cache
                    Q_new, R_new, exist_freq = qr_cache[new_M]
                    qr_cache[new_M] = (Q_new, R_new, exist_freq + 1)
                else:
                    # Use optimized QR update (only if beneficial for performance)
                    if len(M) > 100:  # Only for large feature sets where QR update is faster than recomputation
                        # Find the new feature (the one not in M)
                        new_feature = best_feature
                        
                        # Fast rank-one update for QR
                        x_new = X_centered[:, new_feature]
                        r_k = Q.T @ x_new
                        q_k_plus_1 = x_new - Q @ r_k
                        q_k_plus_1_norm = np.linalg.norm(q_k_plus_1)
                        
                        if q_k_plus_1_norm > 1e-10:
                            q_k_plus_1 = q_k_plus_1 / q_k_plus_1_norm
                            
                            # Extend Q and R
                            Q_new = np.column_stack([Q, q_k_plus_1])
                            
                            R_new = np.zeros((Q_new.shape[1], Q_new.shape[1]))
                            R_new[:R.shape[0], :R.shape[1]] = R
                            R_new[:R.shape[0], -1] = r_k
                            R_new[-1, -1] = q_k_plus_1_norm
                            
                            qr_cache[new_M] = (Q_new, R_new, 1)
        
        return coef_k, new_sets
    
    def _fit_specialized_forward_selection(self, X, y):
        """
        Highly optimized implementation for forward selection (m=p, B=1).
        Uses vectorized operations and true rank-one QR updates.
        """
        n_samples, n_features = X.shape
        
        # Center data (one-time operation)
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean
        
        # Initialize storage for models
        self.coef_ = [np.zeros(n_features)]
        self.intercept_ = [y_mean]
        
        # Initialize tracking
        selected = []
        unselected = np.arange(n_features)
        
        # Pre-allocate matrices for QR factorization
        # Allocate the maximum size we might need
        Q_full = np.zeros((n_samples, self.k_max))
        R_full = np.zeros((self.k_max, self.k_max))
        
        # Keep track of actual size of QR factorization
        k_current = 0
        
        # Initial residuals = y_centered (since no features are selected yet)
        residuals = y_centered.copy()
        
        # Pre-compute feature norms (constant across iterations)
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        feature_norms[feature_norms < 1e-10] = 1.0
        
        # Main loop - select one feature at a time
        for k in range(1, self.k_max + 1):
            if len(unselected) == 0:
                # No features left, just break out of the loop
                break
            
            # Compute selection criterion efficiently
            if k == 1:
                # First feature - just use |X^T y| / ||X||
                X_unsel = X_centered[:, unselected]
                correlations = np.abs(X_unsel.T @ y_centered)
                sel_norms = feature_norms[unselected]
                selection_values = correlations / sel_norms
            else:
                # For subsequent features
                # Extract unselected features
                X_unsel = X_centered[:, unselected]
                
                # Current Q matrix
                Q = Q_full[:, :k_current]
                
                # 1. Compute correlations: |X_unsel^T residuals|
                correlations = np.abs(X_unsel.T @ residuals)
                
                # 2. Compute orthogonal components efficiently
                # X_orth = X_unsel - Q(Q^T X_unsel)
                QTX = Q.T @ X_unsel
                X_orth = X_unsel - Q @ QTX
                
                # Compute norms of orthogonal components
                orth_norms = np.sqrt(np.sum(X_orth**2, axis=0))
                
                # Handle numerical stability
                mask = orth_norms < 1e-10
                orth_norms[mask] = np.inf  # Avoid division by zero
                
                # Compute selection values
                selection_values = correlations / orth_norms
            
            # Find best feature
            best_idx = np.argmax(selection_values)
            best_feature = unselected[best_idx]
            
            # Update tracking
            selected.append(best_feature)
            unselected = np.delete(unselected, best_idx)
            
            # Get the selected feature vector
            x_best = X_centered[:, best_feature]
            
            # Update QR factorization with rank-one update
            if k_current == 0:
                # First feature - just normalize and store
                norm_x = np.linalg.norm(x_best)
                Q_full[:, 0] = x_best / norm_x
                R_full[0, 0] = norm_x
                k_current = 1
            else:
                # True rank-one update
                
                # Current Q matrix
                Q = Q_full[:, :k_current]
                
                # 1. Project x_best onto current Q
                QTx = Q.T @ x_best
                
                # 2. Compute orthogonal component
                x_orth = x_best - Q @ QTx
                orth_norm = np.linalg.norm(x_orth)
                
                # 3. Update Q and R if orthogonal component is significant
                if orth_norm > 1e-10:
                    # Normalize orthogonal component
                    Q_full[:, k_current] = x_orth / orth_norm
                    
                    # Update R
                    R_full[:k_current, k_current] = QTx
                    R_full[k_current, k_current] = orth_norm
                    
                    # Increment the size counter
                    k_current += 1
            
            # Update residuals: r = y - Q(Q^T y)
            Q = Q_full[:, :k_current]
            QTy = Q.T @ y_centered
            residuals = y_centered - Q @ QTy
            
            # Compute coefficients using R
            R = R_full[:k_current, :k_current]
            beta = linalg.solve_triangular(R, QTy, lower=False, check_finite=False)
            
            # Create coefficient vector
            coef_k = np.zeros(n_features)
            coef_k[selected] = beta
            
            # Store model
            self.coef_.append(coef_k)
            self.intercept_.append(y_mean)
        
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