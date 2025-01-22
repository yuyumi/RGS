from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from scipy.linalg import lstsq
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class RGS(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, alpha=None, m=None, n_estimators=1000, n_resample_iter=0, random_state=None):
        self._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m  # Keep m as None if alpha is provided
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Validate and initialize
        X, y = self._validate_training_inputs(X, y)
        _, self.p = X.shape
        
        # Center and scale X efficiently
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - X_mean
        norms = np.sqrt(np.sum(X_centered ** 2, axis=0, keepdims=True))
        norms[norms == 0] = 1
        X_scaled = X_centered / norms
        
        # Initialize parameters
        generator = np.random.default_rng(self.random_state)
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] += Counter({frozenset({}): self.n_estimators})
        self.coef_ = np.zeros((self.k_max + 1, self.p))
        self.intercept_ = np.zeros(self.k_max + 1)
        y_mean = np.mean(y)
        
        # Main loop
        for k in range(self.k_max + 1):
            Ms = np.array(list(self.feature_sets[k].keys()))
            freqs = np.array(list(self.feature_sets[k].values()))
            
            # Resample if needed
            if self.n_resample_iter > 0:
                proportions = freqs / self.n_estimators
                freqs = generator.multinomial(self.n_estimators, proportions)
                mask = freqs > 0
                Ms = Ms[mask]
                freqs = freqs[mask]
            
            # Process each feature set
            coef_ = np.zeros(self.p)
            for M, freq in zip(Ms, freqs):
                M = list(M)
                beta, _, _, _ = lstsq(X_centered[:, M], y)
                residuals = y - X_centered[:, M] @ beta
                coef_[M] += beta * freq
                
                if k < self.k_max:
                    # Get complement features efficiently
                    mask = np.ones(self.p, dtype=bool)
                    mask[M] = False
                    M_comp = np.arange(self.p)[mask]
                    
                    # Compute correlations and update feature sets
                    correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                    self.feature_sets[k + 1] += self._get_new_feature_sets(
                        M, M_comp, correlations, freq, generator)
            
            # Update parameters
            self.coef_[k] = coef_ / self.n_estimators
            self.intercept_[k] = y_mean - np.dot(X_mean.ravel(), self.coef_[k])
        
        return self

    def predict(self, X, k=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if k is None:
            k = self.k_max
        return X @ self.coef_[k] + self.intercept_[k]
    
    def _get_new_feature_sets(self, M, M_comp, correlations, n_iter, generator):
        if len(M_comp) == 0:
            return Counter()
        
        # Always use all remaining features when m exceeds available features
        n_candidates = min(self.m, len(M_comp))
        
        # Generate candidates one iteration at a time to avoid sampling issues
        candidates = []
        for _ in range(n_iter):
            candidates.append(generator.choice(
                range(len(M_comp)),
                size=n_candidates,
                replace=False
            ))
        candidates = np.array(candidates)
        
        candidate_correlations = correlations[candidates]
        max_index_in_subset = np.argmax(candidate_correlations, axis=1)
        psi_vals = M_comp[candidates[range(n_iter), max_index_in_subset]]
        
        psi_freqs = np.bincount(psi_vals)
        psi_vals_unique = np.nonzero(psi_freqs)[0]
        M_new_unique = [frozenset(set(M) | {feat}) for feat in psi_vals_unique]
        
        return Counter(dict(zip(M_new_unique, psi_freqs[psi_vals_unique])))

    @staticmethod
    def _validate_args(k, alpha, m, n_estimators):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("k must be a positive integer")
        if not (isinstance(n_estimators, int) and n_estimators > 0):
            raise ValueError("n_estimators must be a positive integer")
        if not (alpha is None or m is None):
            raise ValueError("Either alpha or m must be provided, not both")
        if alpha is not None and not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if m is not None:
            try:
                m = int(m)
                if m <= 0:
                    raise ValueError("m must be a positive integer")
            except (TypeError, ValueError):
                raise ValueError("m must be a positive integer")

    def _validate_training_inputs(self, X, y):
        X, y = check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        p = X.shape[1]
        if self.k_max > p:
            raise ValueError("k_max cannot be larger than number of features")
            
        # Handle m calculation
        if self.m is None:
            if self.alpha is None:
                raise ValueError("Either alpha or m must be provided")
            self.m = int(np.ceil(self.alpha * p))
        else:
            self.m = int(self.m)
            
        if self.m > p:
            raise ValueError("m cannot be larger than number of features")
            
        return X, y


class RGSCV(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, m_grid, n_estimators=1000, n_resample_iter=0, random_state=None, cv=5):
        self._validate_params(k_max, m_grid, n_estimators, n_resample_iter, cv)
        self.k_max = int(k_max)  # Ensure k_max is integer
        self.m_grid = np.array([int(m) for m in m_grid])  # Convert grid values to integers
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state
        self.cv = cv
        
    def fit(self, X, y):
        # Input validation and conversion
        X, y = self._validate_data(X, y)
        
        # Initialize scores array
        n_m_values = len(self.m_grid)
        n_k_values = self.k_max
        cv_splitter = self._get_cv_splitter()
        n_splits = cv_splitter.get_n_splits(X)
        
        # Pre-allocate scores array [k, m, fold]
        self.cv_scores_ = np.full((n_k_values, n_m_values, n_splits), np.nan)
        
        # Perform CV
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Try each m value
            for m_idx, m in enumerate(self.m_grid):
                model = RGS(
                    k_max=int(self.k_max),  # Ensure k_max is int
                    m=int(m),  # Ensure m is int
                    n_estimators=self.n_estimators,
                    n_resample_iter=self.n_resample_iter,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
                
                # Evaluate for each k efficiently
                for k in range(1, self.k_max + 1):
                    y_pred = model.predict(X_val, k=k)
                    self.cv_scores_[k-1, m_idx, fold_idx] = mean_squared_error(y_val, y_pred)
        
        # Find best parameters efficiently using numpy operations
        mean_scores = np.nanmean(self.cv_scores_, axis=2)  # Average across folds
        best_scores = np.min(mean_scores, axis=1)  # Best score for each k
        best_m_indices = np.argmin(mean_scores, axis=1)  # Best m index for each k
        
        # Find optimal k
        self.k_ = np.argmin(best_scores) + 1
        self.m_ = self.m_grid[best_m_indices[self.k_ - 1]]
        
        # Store best scores and parameters
        self.best_scores_ = {
            k + 1: {
                'm': self.m_grid[best_m_indices[k]],
                'score': best_scores[k]
            }
            for k in range(n_k_values)
        }
        
        # Fit final model with best parameters
        self.model_ = RGS(
            k_max=int(self.k_),
            m=int(self.m_),  # Ensure m is int
            n_estimators=self.n_estimators,
            n_resample_iter=self.n_resample_iter,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        # Copy attributes from final model for direct access
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self.feature_sets = self.model_.feature_sets
        self.p = self.model_.p
        
        return self
    
    def predict(self, X):
        """Predict using the model with best parameters."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X @ self.coef_[self.k_] + self.intercept_[self.k_]
    
    def _get_cv_splitter(self):
        """Get CV splitter object."""
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return self.cv
    
    def _validate_data(self, X, y):
        """Validate and convert input data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X, y = check_X_y(X, y)
        
        if X.shape[1] < self.k_max:
            raise ValueError("k_max cannot be larger than number of features")
        if any(m > X.shape[1] for m in self.m_grid):
            raise ValueError("m values cannot be larger than number of features")
            
        return X, y
    
    def _validate_params(self, k_max, m_grid, n_estimators, n_resample_iter, cv):
        """Validate input parameters."""
        if not isinstance(k_max, int) or k_max <= 0:
            raise ValueError("k_max must be a positive integer")
            
        try:
            m_grid = [int(m) for m in m_grid]
            if not all(m > 0 for m in m_grid):
                raise ValueError("all m values must be positive integers")
        except (TypeError, ValueError):
            raise ValueError("all m values must be convertible to positive integers")
            
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
            
        if not isinstance(n_resample_iter, int) or n_resample_iter < 0:
            raise ValueError("n_resample_iter must be a non-negative integer")
            
        if isinstance(cv, int) and cv <= 1:
            raise ValueError("cv must be greater than 1")

class EnhancedRGS(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, alpha=None, m=None, n_estimators=1000, n_resample_iter=0, batch_size=100, random_state=None):
        self._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.batch_size = min(batch_size, n_estimators)
        self.random_state = random_state

    def fit(self, X, y):
        # Validate and initialize
        X, y = self._validate_training_inputs(X, y)
        _, self.p = X.shape
        
        # Initialize random state
        rng = np.random.default_rng(self.random_state)
        
        # Precompute common matrices
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - X_mean
        X_norm = np.sqrt(np.sum(X_centered ** 2, axis=0, keepdims=True))
        X_norm[X_norm == 0] = 1
        X_scaled = X_centered / X_norm
        y_mean = np.mean(y)
        
        # Initialize storage
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] = Counter({frozenset(): self.n_estimators})
        self.coef_ = np.zeros((self.k_max + 1, self.p))
        self.intercept_ = np.zeros(self.k_max + 1)
        
        # Precompute initial correlations
        initial_correlations = np.abs(X_scaled.T @ y)
        
        # Main loop
        for k in range(self.k_max + 1):
            features_coef = np.zeros(self.p)
            current_sets = list(self.feature_sets[k].items())
            
            # Handle resampling if needed
            if self.n_resample_iter > 0:
                # Convert current sets to numpy arrays for efficient operations
                Ms = np.array([list(M) for M, _ in current_sets], dtype=object)
                freqs = np.array([freq for _, freq in current_sets])
                proportions = freqs / self.n_estimators
                
                # Resample frequencies using multinomial distribution
                freqs = rng.multinomial(self.n_estimators, proportions)
                mask = freqs > 0
                Ms = Ms[mask]
                freqs = freqs[mask]
                
                # Update current_sets with resampled values
                current_sets = [(frozenset(M), freq) for M, freq in zip(Ms, freqs)]
            
            # Process feature sets
            for M_frozen, freq in current_sets:
                M = list(M_frozen)
                
                if M:
                    # Efficient matrix computations for current feature set
                    X_M = X_centered[:, M]
                    beta = lstsq(X_M, y)[0]
                    features_coef[M] += beta * freq
                    
                    if k < self.k_max:
                        # Get complement features efficiently
                        mask = np.ones(self.p, dtype=bool)
                        mask[M] = False
                        M_comp = np.arange(self.p)[mask]
                        
                        if len(M_comp) > 0:
                            # Compute residuals and correlations
                            predicted = X_M @ beta
                            residuals = y - predicted
                            correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                            
                            # Process in smaller batches
                            remaining_freq = freq
                            while remaining_freq > 0:
                                current_batch = min(self.batch_size, remaining_freq)
                                n_candidates = min(self.m, len(M_comp))
                                
                                # Generate and process candidates
                                for _ in range(current_batch):
                                    candidates = rng.choice(
                                        len(M_comp),
                                        size=n_candidates,
                                        replace=False
                                    )
                                    best_idx = candidates[np.argmax(correlations[candidates])]
                                    new_feature = M_comp[best_idx]
                                    new_set = frozenset(set(M) | {new_feature})
                                    self.feature_sets[k + 1][new_set] += 1
                                
                                remaining_freq -= current_batch
                else:
                    # Handle empty set case
                    if k < self.k_max:
                        remaining_freq = freq
                        while remaining_freq > 0:
                            current_batch = min(self.batch_size, remaining_freq)
                            n_candidates = min(self.m, self.p)
                            
                            # Process one at a time to avoid sampling issues
                            for _ in range(current_batch):
                                candidates = rng.choice(
                                    self.p,
                                    size=n_candidates,
                                    replace=False
                                )
                                best_idx = np.argmax(initial_correlations[candidates])
                                new_feature = candidates[best_idx]
                                new_set = frozenset({new_feature})
                                self.feature_sets[1][new_set] += 1
                            
                            remaining_freq -= current_batch
            
            # Update coefficients
            self.coef_[k] = features_coef / self.n_estimators
            self.intercept_[k] = y_mean - np.dot(X_mean.ravel(), self.coef_[k])
        
        return self

    def predict(self, X, k=None):
        if k is None:
            k = self.k_max
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X @ self.coef_[k] + self.intercept_[k]
    
    @staticmethod
    def _validate_args(k, alpha, m, n_estimators):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("k must be a positive integer")
        if not (isinstance(n_estimators, int) and n_estimators > 0):
            raise ValueError("n_estimators must be a positive integer")
        if not (alpha is None or m is None):
            raise ValueError("Either alpha or m must be provided, not both")
        if alpha is not None and not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if m is not None:
            try:
                m = int(m)
                if m <= 0:
                    raise ValueError("m must be a positive integer")
            except (TypeError, ValueError):
                raise ValueError("m must be a positive integer")

    def _validate_training_inputs(self, X, y):
        X, y = check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        p = X.shape[1]
        if self.k_max > p:
            raise ValueError("k_max cannot be larger than number of features")
            
        if self.m is None:
            if self.alpha is None:
                raise ValueError("Either alpha or m must be provided")
            self.m = int(np.ceil(self.alpha * p))
        else:
            self.m = int(self.m)
            
        if self.m > p:
            raise ValueError("m cannot be larger than number of features")
            
        return X, y

class EnhancedRGS(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, alpha=None, m=None, n_estimators=1000, n_resample_iter=0, batch_size=100, random_state=None):
        self._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.batch_size = min(batch_size, n_estimators)
        self.random_state = random_state

    def fit(self, X, y):
        # Validate and initialize
        X, y = self._validate_training_inputs(X, y)
        _, self.p = X.shape
        
        # Initialize random state
        rng = np.random.default_rng(self.random_state)
        
        # Precompute common matrices
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - X_mean
        X_norm = np.sqrt(np.sum(X_centered ** 2, axis=0, keepdims=True))
        X_norm[X_norm == 0] = 1
        X_scaled = X_centered / X_norm
        y_mean = np.mean(y)
        
        # Initialize storage
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] = Counter({frozenset(): self.n_estimators})
        self.coef_ = np.zeros((self.k_max + 1, self.p))
        self.intercept_ = np.zeros(self.k_max + 1)
        
        # Precompute initial correlations
        initial_correlations = np.abs(X_scaled.T @ y)
        
        # Main loop
        for k in range(self.k_max + 1):
            features_coef = np.zeros(self.p)
            current_sets = list(self.feature_sets[k].items())
            
            # Handle resampling if needed
            if self.n_resample_iter > 0:
                # Convert current sets to numpy arrays for efficient operations
                Ms = np.array([list(M) for M, _ in current_sets], dtype=object)
                freqs = np.array([freq for _, freq in current_sets])
                proportions = freqs / self.n_estimators
                
                # Resample frequencies using multinomial distribution
                freqs = rng.multinomial(self.n_estimators, proportions)
                mask = freqs > 0
                Ms = Ms[mask]
                freqs = freqs[mask]
                
                # Update current_sets with resampled values
                current_sets = [(frozenset(M), freq) for M, freq in zip(Ms, freqs)]
            
            # Process feature sets
            for M_frozen, freq in current_sets:
                M = list(M_frozen)
                
                if M:
                    # Efficient matrix computations for current feature set
                    X_M = X_centered[:, M]
                    beta = lstsq(X_M, y)[0]
                    features_coef[M] += beta * freq
                    
                    if k < self.k_max:
                        # Get complement features efficiently
                        mask = np.ones(self.p, dtype=bool)
                        mask[M] = False
                        M_comp = np.arange(self.p)[mask]
                        
                        if len(M_comp) > 0:
                            # Compute residuals and correlations
                            predicted = X_M @ beta
                            residuals = y - predicted
                            correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                            
                            # Process in smaller batches
                            remaining_freq = freq
                            while remaining_freq > 0:
                                current_batch = min(self.batch_size, remaining_freq)
                                n_candidates = min(self.m, len(M_comp))
                                
                                # Generate and process candidates
                                for _ in range(current_batch):
                                    candidates = rng.choice(
                                        len(M_comp),
                                        size=n_candidates,
                                        replace=False
                                    )
                                    best_idx = candidates[np.argmax(correlations[candidates])]
                                    new_feature = M_comp[best_idx]
                                    new_set = frozenset(set(M) | {new_feature})
                                    self.feature_sets[k + 1][new_set] += 1
                                
                                remaining_freq -= current_batch
                else:
                    # Handle empty set case
                    if k < self.k_max:
                        remaining_freq = freq
                        while remaining_freq > 0:
                            current_batch = min(self.batch_size, remaining_freq)
                            n_candidates = min(self.m, self.p)
                            
                            # Process one at a time to avoid sampling issues
                            for _ in range(current_batch):
                                candidates = rng.choice(
                                    self.p,
                                    size=n_candidates,
                                    replace=False
                                )
                                best_idx = np.argmax(initial_correlations[candidates])
                                new_feature = candidates[best_idx]
                                new_set = frozenset({new_feature})
                                self.feature_sets[1][new_set] += 1
                            
                            remaining_freq -= current_batch
            
            # Update coefficients
            self.coef_[k] = features_coef / self.n_estimators
            self.intercept_[k] = y_mean - np.dot(X_mean.ravel(), self.coef_[k])
        
        return self

    def predict(self, X, k=None):
        if k is None:
            k = self.k_max
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X @ self.coef_[k] + self.intercept_[k]
    
    @staticmethod
    def _validate_args(k, alpha, m, n_estimators):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("k must be a positive integer")
        if not (isinstance(n_estimators, int) and n_estimators > 0):
            raise ValueError("n_estimators must be a positive integer")
        if not (alpha is None or m is None):
            raise ValueError("Either alpha or m must be provided, not both")
        if alpha is not None and not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if m is not None:
            try:
                m = int(m)
                if m <= 0:
                    raise ValueError("m must be a positive integer")
            except (TypeError, ValueError):
                raise ValueError("m must be a positive integer")

    def _validate_training_inputs(self, X, y):
        X, y = check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        p = X.shape[1]
        if self.k_max > p:
            raise ValueError("k_max cannot be larger than number of features")
            
        if self.m is None:
            if self.alpha is None:
                raise ValueError("Either alpha or m must be provided")
            self.m = int(np.ceil(self.alpha * p))
        else:
            self.m = int(self.m)
            
        if self.m > p:
            raise ValueError("m cannot be larger than number of features")
            
        return X, y

class BatchedFastRGS(BaseEstimator, RegressorMixin):
    """
    Batched Fast Randomized Greedy Selection (BatchedFastRGS) combines the simplified 
    implementation of FastRGS with memory-efficient batching for large datasets.
    
    Parameters
    ----------
    k_max : int
        Maximum number of features to select
    alpha : float, optional (default=None)
        Fraction of features to randomly select as candidates
    m : int, optional (default=None)
        Fixed number of features to randomly select as candidates
    n_estimators : int, default=1000
        Number of models in the ensemble
    batch_size : int, default=100
        Size of batches for processing
    n_resample_iter : int, default=0
        Number of resampling iterations
    random_state : int or None, default=None
        Random number generator seed
    """
    
    def __init__(self, k_max, alpha=None, m=None, n_estimators=1000, 
                 batch_size=100, n_resample_iter=0, random_state=None):
        self._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_estimators = n_estimators
        self.batch_size = min(batch_size, n_estimators)
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state

    def fit(self, X, y):
        # Validate and preprocess data
        X, y = self._validate_training_inputs(X, y)
        
        # Initialize RNG and data structures
        rng = np.random.default_rng(self.random_state)
        _, self.p = X.shape
        
        # Precompute common matrices
        X_centered = X - X.mean(axis=0)
        X_norms = np.sqrt(np.sum(X_centered ** 2, axis=0))
        X_norms[X_norms == 0] = 1  # Avoid division by zero
        X_scaled = X_centered / X_norms
        
        # Initialize storage
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] = Counter({frozenset(): self.n_estimators})
        self.coef_ = []
        self.intercept_ = []
        
        # Precompute initial correlations for empty set case
        y_centered = y - y.mean()
        initial_correlations = np.abs(X_scaled.T @ y_centered)
        
        # Main loop over feature set sizes
        for k in range(self.k_max + 1):
            Ms = list(self.feature_sets[k].items())
            coef_ = np.zeros(self.p)
            
            # Resample if needed
            if self.n_resample_iter > 0:
                proportions = np.array([freq for _, freq in Ms]) / self.n_estimators
                freqs = rng.multinomial(self.n_estimators, proportions)
                Ms = [(M, freq) for (M, _), freq in zip(Ms, freqs) if freq > 0]
            
            # Process each feature set
            for M_frozen, freq in Ms:
                M = list(M_frozen) if M_frozen else []
                
                if M:
                    # Process non-empty feature sets
                    X_M = X_centered[:, M]
                    beta = lstsq(X_M, y)[0]
                    coef_[M] += beta * freq
                    
                    if k < self.k_max:
                        # Get complement features
                        mask = np.ones(self.p, dtype=bool)
                        mask[M] = False
                        M_comp = np.arange(self.p)[mask]
                        
                        # Compute residuals and correlations
                        residuals = y - X_M @ beta
                        correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                        
                        # Process in batches
                        self._process_batches(M, M_comp, correlations, freq, k, rng)
                else:
                    # Handle empty set case
                    if k < self.k_max:
                        self._process_empty_set(initial_correlations, freq, k, rng)
            
            # Update model parameters
            self.coef_.append(coef_ / self.n_estimators)
            self.intercept_.append(y.mean() - np.dot(X.mean(axis=0), coef_ / self.n_estimators))
        
        return self

    def _process_batches(self, M, M_comp, correlations, total_freq, k, rng):
        """Process feature selection in batches to manage memory usage."""
        remaining_freq = total_freq
        n_candidates = min(self.m, len(M_comp))
        
        while remaining_freq > 0:
            current_batch = min(self.batch_size, remaining_freq)
            
            # Generate candidates for current batch
            candidates = np.zeros((current_batch, n_candidates), dtype=int)
            for i in range(current_batch):
                candidates[i] = rng.choice(len(M_comp), size=n_candidates, replace=False)
            
            # Find best features
            candidate_correlations = correlations[candidates]
            max_indices = np.argmax(candidate_correlations, axis=1)
            selected_features = M_comp[candidates[range(current_batch), max_indices]]
            
            # Update feature sets
            for feature in selected_features:
                new_set = frozenset(set(M) | {feature})
                self.feature_sets[k + 1][new_set] += 1
            
            remaining_freq -= current_batch

    def _process_empty_set(self, correlations, total_freq, k, rng):
        """Process empty set case in batches."""
        remaining_freq = total_freq
        n_candidates = min(self.m, self.p)
        
        while remaining_freq > 0:
            current_batch = min(self.batch_size, remaining_freq)
            
            # Generate and process candidates
            for _ in range(current_batch):
                candidates = rng.choice(self.p, size=n_candidates, replace=False)
                best_idx = np.argmax(correlations[candidates])
                new_feature = candidates[best_idx]
                new_set = frozenset({new_feature})
                self.feature_sets[k + 1][new_set] += 1
            
            remaining_freq -= current_batch

    def predict(self, X, k=None):
        """Make predictions using the model at step k."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        assert X.ndim == 2 and X.shape[1] == self.p
        if k is None:
            k = self.k_max
        return X @ self.coef_[k] + self.intercept_[k]

    @staticmethod
    def _validate_args(k, alpha, m, n_estimators):
        """Validate input arguments."""
        assert isinstance(k, int) and k > 0
        assert isinstance(n_estimators, int) and n_estimators > 0
        assert alpha is None or m is None, "Provide either alpha or m, not both"
        if alpha is not None:
            assert 0 < alpha <= 1
        if m is not None:
            assert isinstance(m, int) and m > 0

    def _validate_training_inputs(self, X, y):
        """Validate training data inputs."""
        X, y = check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        _, p = X.shape
        assert self.k_max <= p, "k_max cannot exceed number of features"
        
        if self.m is None:
            self.m = int(np.ceil(self.alpha * p))
        assert self.m <= p, "m cannot exceed number of features"
        
        return X, y
