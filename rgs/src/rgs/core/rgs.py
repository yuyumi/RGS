from collections import Counter
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from scipy import linalg
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from collections import defaultdict, Counter

import numba as nb
from numba import jit, float64, int64, boolean

@jit(nopython=True)
def orthogonalize_vector(x, Q):
    """
    Orthogonalize vector x against columns of Q using vectorized operations.
    This is much faster than the loop-based approach, especially for larger Q.
    
    Mathematical operation: x_orth = x - Q @ (Q.T @ x)
    """
    if Q.shape[1] == 0:
        # No vectors to orthogonalize against
        return x.copy()
    
    # Vectorized version: compute all projections at once using BLAS
    # This is equivalent to the loop but much faster
    projections = Q.T @ x  # Shape: (k,) where k = Q.shape[1]
    x_orth = x - Q @ projections  # Vectorized subtraction of all projections
    
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
             n_resample_iter=1, method='fs', random_state=None):
        self.k_max = k_max
        self.m = m
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.method = method  # New parameter
        self.random_state = random_state
        
    def _validate_inputs(self, X, y):
        """Validate input data and parameters."""
        n_samples, n_features = X.shape
        
        # Validate feature selection parameters
        assert self.k_max > 0, "k_max must be positive"
        assert self.k_max <= n_features, "k_max cannot exceed number of features"
        
        # Validate method parameter
        assert self.method in ['fs', 'omp'], "method must be 'fs' or 'omp'"
        
        # Check for near-constant features
        X_centered = X - X.mean(axis=0)
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        
        near_zero_features = np.where(feature_norms < 1e-10)[0]
        if len(near_zero_features) > 0:
            raise ValueError(
                f"Features {list(near_zero_features)} have near-zero variance "
                f"(norms < 1e-10). These are essentially constant features that "
                f"provide no predictive information and cause numerical instability. "
                f"Please remove them before fitting using sklearn.feature_selection.VarianceThreshold "
                f"or manual preprocessing."
            )
        
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
        
        # Always use dictionary-based implementation (more reliable and general)
        if n_features > 100:
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
    
    def _compute_selection_criterion(self, correlations, norms, method=None):
        """
        Compute selection criterion based on method.
        
        Parameters
        ----------
        correlations : array
            Absolute correlations |X^T * residual|
        norms : array  
            Norms of features (for FS) or orthogonal components (for subsequent features)
        method : str, optional
            Method to use. If None, uses self.method
        
        Returns
        -------
        selection_values : array
            Selection criterion values
        """
        if method is None:
            method = self.method
            
        if method == 'fs':
            # Forward Selection: normalize by norms
            valid_mask = norms > 1e-10
            result = np.full_like(correlations, -np.inf)
            result[valid_mask] = correlations[valid_mask] / norms[valid_mask]
            return result
        elif method == 'omp':
            # OMP: use raw correlations
            return correlations
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _vectorized_bootstrap_resample(self, feature_freqs, bootstrap_iter=0):
        """
        Highly optimized bootstrap resampling using vectorized operations.
        Dictionary-based implementation only.
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
            fs_values = self._compute_selection_criterion(
    correlations, feature_norms[unique_candidates])
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
            if np.any(valid_mask):
                fs_values[valid_mask] = self._compute_selection_criterion(
        correlations[valid_mask], orth_norms[valid_mask])
        
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
        new_sets = defaultdict(int)  # Use defaultdict for cleaner counting
        
        # Sort feature sets by size for better cache locality
        sorted_sets = sorted(active_sets.items(), key=lambda x: (len(x[0]), x[0]))
        
        # First pass: Compute QR, coefficients, and residuals for all feature sets
        all_data = []  # Will store (M, freq, Q, residuals, M_comp)
        
        for M, freq in sorted_sets:
            # Get or compute QR (simplified - no frequency tracking)
            if M in qr_cache:
                Q, R = qr_cache[M]
                del qr_cache[M]  # Delete after use - no frequency tracking needed
            else:
                # Compute QR from scratch
                if len(M) > 0:
                    M_array = np.array(M, dtype=int)
                    X_M = X_centered[:, M_array]
                    Q, R = linalg.qr(X_M, mode='economic', check_finite=False)
                else:
                    Q = np.empty((self.n_samples, 0))
                    R = np.empty((0, 0))
            
            # Compute coefficients
            if len(M) > 0:
                M_array = np.array(M, dtype=int)
                beta = linalg.solve_triangular(R, Q.T @ y_centered, lower=False, check_finite=False)
                coef_k[M_array] += beta * freq
            
            # Compute residuals
            residuals = y_centered - Q @ (Q.T @ y_centered) if len(M) > 0 else y_centered.copy()
            
            # Get unselected features
            unselected_mask = np.ones(self.n_features, dtype=bool)
            if len(M) > 0:
                unselected_mask[list(M)] = False
            M_comp = np.nonzero(unselected_mask)[0]
            
            # Store all data needed for processing candidates
            all_data.append((M, freq, Q, residuals, M_comp))
        
        # Second pass: True batch processing of all candidate features
        for M, freq, Q, residuals, M_comp in all_data:
            if len(M_comp) == 0 or freq == 0:
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
            
            # Generate ALL random samples at once
            all_samples = []
            for i in range(freq):
                sample_indices = generator.choice(len(M_comp), size=n_candidates, replace=False)
                sample = M_comp[sample_indices]
                all_samples.append(sample)
            
            # Get all unique candidates across all samples
            all_candidates = set()
            for sample in all_samples:
                all_candidates.update(sample)
            all_candidates = np.array(list(all_candidates))
            
            # Batch computation of selection criteria for all candidates
            X_candidates = X_centered[:, all_candidates]
            correlations = np.abs(residuals @ X_candidates)
            
            # Vectorized computation of selection criteria
            if Q.shape[1] == 0:
                # First feature - simple correlation/norm
                fs_values = self._compute_selection_criterion(
                    correlations, feature_norms[all_candidates])
            else:
                # Orthogonalization for subsequent features
                proj_matrix = Q.T @ X_candidates
                x_orth = X_candidates - Q @ proj_matrix
                orth_norms = np.linalg.norm(x_orth, axis=0)
                
                # Handle numerical issues
                fs_values = np.full(len(all_candidates), -np.inf)
                valid_mask = orth_norms > 1e-10
                if np.any(valid_mask):
                    fs_values[valid_mask] = self._compute_selection_criterion(
                        correlations[valid_mask], orth_norms[valid_mask])
            
            # Create lookup dictionary for fast access
            criteria_lookup = {feat: val for feat, val in zip(all_candidates, fs_values)}
            
            # TRUE BATCH PROCESSING: Process all samples at once
            if len(all_samples) > 0:
                # Vectorized lookup of selection values for all samples
                all_sample_values = np.array([[criteria_lookup[feat] for feat in sample] 
                                            for sample in all_samples])
                
                # Vectorized selection of best features
                best_indices = np.argmax(all_sample_values, axis=1)
                best_features = [all_samples[i][best_idx] for i, best_idx in enumerate(best_indices)]
                
                # Create all new feature sets at once
                M_list = list(M)
                new_feature_sets_list = [tuple(sorted(M_list + [feat])) for feat in best_features]
                
                # Count frequencies efficiently using Counter
                new_counts = Counter(new_feature_sets_list)
                
                # Merge with existing new_sets using defaultdict
                for feature_set, count in new_counts.items():
                    new_sets[feature_set] += count
        
        return coef_k, dict(new_sets)  # Convert back to regular dict for compatibility
    
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
        
        # Optimize memory layout for better cache performance (safe approach)
        if not X_centered.flags['C_CONTIGUOUS']:
            X_centered = np.ascontiguousarray(X_centered)
        if not y_centered.flags['C_CONTIGUOUS']:
            y_centered = np.ascontiguousarray(y_centered)
        
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
        # feature_norms[feature_norms < 1e-10] = 1.0
        
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
                selection_values = self._compute_selection_criterion(correlations, sel_norms)
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
                selection_values = self._compute_selection_criterion(correlations, orth_norms)
            
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
        if self.m == self.n_features:
            return self._fit_specialized_forward_selection(X, y)
        
        # Initialize data with vectorized operations
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean
        
        # Optimize memory layout for better cache performance (safe approach)
        if not X_centered.flags['C_CONTIGUOUS']:
            X_centered = np.ascontiguousarray(X_centered)
        if not y_centered.flags['C_CONTIGUOUS']:
            y_centered = np.ascontiguousarray(y_centered)
        
        # Pre-compute feature norms using vectorized operations
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        # feature_norms[feature_norms < 1e-10] = 1.0
        
        # Initialize feature tracking - dictionary-based implementation
        feature_sets = [{} for _ in range(self.k_max + 2)]  # +2 to avoid index errors
        feature_sets[0][tuple()] = self.n_estimators
        
        # Initialize model storage
        self.coef_ = []
        self.intercept_ = []
        
        # QR cache
        qr_cache = {}
        qr_cache[tuple()] = (np.empty((self.n_samples, 0)), np.empty((0, 0)))
        
        # Main loop over feature counts
        for k in range(self.k_max + 1):
            # print(f"Processing k={k}")
            
            # Get current feature sets
            active_sets = feature_sets[k].copy()
            
            # Apply bootstrap resampling (with fixed strength - faithful to paper)
            if k > 0 and self.n_resample_iter > 0:
                for iter_idx in range(self.n_resample_iter):
                    active_sets = self._vectorized_bootstrap_resample(
                        active_sets, iter_idx)
            
            # Process feature sets in batches
            coef_k, new_sets = self._process_feature_sets_in_batches(
                X_centered, y_centered, active_sets, feature_norms, qr_cache)
            
            # Update feature sets for next iteration - ONLY if not at max features
            if k < self.k_max:
                # Update dictionary
                feature_sets[k+1] = new_sets
            
            # Compute final coefficients
            self.coef_.append(coef_k / self.n_estimators)
            self.intercept_.append(y_mean - np.dot(X_mean, coef_k / self.n_estimators))
            
            # Memory optimization: clear unnecessary data
            if k >= 2:
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
             k_grid=None, cv=5, scoring=None, method='fs', random_state=None):
        self.k_max = k_max
        self.m_grid = m_grid
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.k_grid = range(1, k_max+1) if k_grid is None else k_grid
        self.method = method
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
                    method=self.method,
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
                        method=self.method,
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
            method=self.method,
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