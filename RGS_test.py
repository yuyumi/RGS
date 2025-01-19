from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from scipy.linalg import lstsq
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class RGS(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, alpha=None, m=None, n_replications=1000, n_resample_iter=0, random_state=None):
        self._validate_args(k_max, alpha, m, n_replications)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m  # Keep m as None if alpha is provided
        self.n_replications = n_replications
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
        self.feature_sets[0] += Counter({frozenset({}): self.n_replications})
        self.coef_ = np.zeros((self.k_max + 1, self.p))
        self.intercept_ = np.zeros(self.k_max + 1)
        y_mean = np.mean(y)
        
        # Main loop
        for k in range(self.k_max + 1):
            Ms = np.array(list(self.feature_sets[k].keys()))
            freqs = np.array(list(self.feature_sets[k].values()))
            
            # Resample if needed
            if self.n_resample_iter > 0:
                proportions = freqs / self.n_replications
                freqs = generator.multinomial(self.n_replications, proportions)
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
            self.coef_[k] = coef_ / self.n_replications
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
    def _validate_args(k, alpha, m, n_replications):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("k must be a positive integer")
        if not (isinstance(n_replications, int) and n_replications > 0):
            raise ValueError("n_replications must be a positive integer")
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
    def __init__(self, k_max, m_grid, n_replications=1000, n_resample_iter=0, random_state=None, cv=5):
        self._validate_params(k_max, m_grid, n_replications, n_resample_iter, cv)
        self.k_max = int(k_max)  # Ensure k_max is integer
        self.m_grid = np.array([int(m) for m in m_grid])  # Convert grid values to integers
        self.n_replications = n_replications
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
                    n_replications=self.n_replications,
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
            n_replications=self.n_replications,
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
    
    def _validate_params(self, k_max, m_grid, n_replications, n_resample_iter, cv):
        """Validate input parameters."""
        if not isinstance(k_max, int) or k_max <= 0:
            raise ValueError("k_max must be a positive integer")
            
        try:
            m_grid = [int(m) for m in m_grid]
            if not all(m > 0 for m in m_grid):
                raise ValueError("all m values must be positive integers")
        except (TypeError, ValueError):
            raise ValueError("all m values must be convertible to positive integers")
            
        if not isinstance(n_replications, int) or n_replications <= 0:
            raise ValueError("n_replications must be a positive integer")
            
        if not isinstance(n_resample_iter, int) or n_resample_iter < 0:
            raise ValueError("n_resample_iter must be a non-negative integer")
            
        if isinstance(cv, int) and cv <= 1:
            raise ValueError("cv must be greater than 1")

class OptimizedRGS(BaseEstimator, RegressorMixin):
    def __init__(self, k_max, alpha=None, m=None, n_replications=1000, n_resample_iter=0, random_state=None):
        self._validate_args(k_max, alpha, m, n_replications)
        self.k_max = k_max
        self.alpha = alpha
        self.m = m
        self.n_replications = n_replications
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        # Validate and initialize
        X, y = self._validate_training_inputs(X, y)
        _, self.p = X.shape
        
        # Use the same random state handling as original
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Center and scale X exactly as in original
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean.reshape(1, -1)
        norms = np.sqrt(np.sum(X_centered ** 2, axis=0))
        norms[norms == 0] = 1  # Avoid division by zero
        X_scaled = X_centered / norms.reshape(1, -1)
        
        # Initialize exactly as in original
        self.feature_sets = [Counter() for _ in range(self.k_max + 1)]
        self.feature_sets[0] = Counter({frozenset(): self.n_replications})  # Empty set
        self.coef_ = np.zeros((self.k_max + 1, self.p))
        self.intercept_ = np.zeros(self.k_max + 1)
        y_mean = np.mean(y)
        
        # Main loop following original logic exactly
        for k in range(self.k_max + 1):
            # Get current feature sets
            current_sets = list(self.feature_sets[k].items())
            
            # Handle resampling exactly as original
            if self.n_resample_iter > 0:
                feature_sets = [fs for fs, _ in current_sets]
                freqs = np.array([freq for _, freq in current_sets])
                proportions = freqs / self.n_replications
                freqs = np.random.multinomial(self.n_replications, proportions)
                current_sets = [(fs, freq) for fs, freq in zip(feature_sets, freqs) if freq > 0]
            
            # Process each feature set
            coef_ = np.zeros(self.p)
            for M_frozen, freq in current_sets:
                M = sorted(list(M_frozen))  # Ensure consistent ordering
                
                if M:  # Non-empty set
                    # Compute regression exactly as original
                    X_M = X_centered[:, M]
                    beta = lstsq(X_M, y)[0]
                    residuals = y - X_M @ beta
                    coef_[M] += beta * freq
                    
                    if k < self.k_max:
                        # Get complement features
                        mask = np.ones(self.p, dtype=bool)
                        mask[M] = False
                        M_comp = np.arange(self.p)[mask]
                        
                        # Compute correlations exactly as original
                        if len(M_comp) > 0:
                            correlations = np.abs(X_scaled[:, M_comp].T @ residuals)
                            
                            # Update feature sets one at a time
                            for _ in range(freq):
                                # Select random candidates
                                n_candidates = min(self.m, len(M_comp))
                                candidates = np.random.choice(
                                    len(M_comp), 
                                    size=n_candidates,
                                    replace=False
                                )
                                
                                # Find best candidate
                                candidate_correlations = correlations[candidates]
                                best_idx = candidates[np.argmax(candidate_correlations)]
                                new_feature = M_comp[best_idx]
                                
                                # Update feature sets
                                new_set = frozenset(set(M) | {new_feature})
                                self.feature_sets[k + 1][new_set] += 1
                else:
                    # Handle empty set case exactly as original
                    if k < self.k_max:
                        M_comp = np.arange(self.p)
                        correlations = np.abs(X_scaled.T @ y)
                        
                        for _ in range(freq):
                            n_candidates = min(self.m, len(M_comp))
                            candidates = np.random.choice(
                                len(M_comp),
                                size=n_candidates,
                                replace=False
                            )
                            
                            candidate_correlations = correlations[candidates]
                            best_idx = candidates[np.argmax(candidate_correlations)]
                            new_feature = M_comp[best_idx]
                            
                            new_set = frozenset({new_feature})
                            self.feature_sets[k + 1][new_set] += 1
            
            # Update coefficients exactly as original
            self.coef_[k] = coef_ / self.n_replications
            self.intercept_[k] = y_mean - np.dot(X_mean, self.coef_[k])
        
        return self

    def predict(self, X, k=None):
        """Make predictions"""
        if k is None:
            k = self.k_max
        if isinstance(X, np.ndarray):
            return X @ self.coef_[k] + self.intercept_[k]
        return X.values @ self.coef_[k] + self.intercept_[k]
    
    @staticmethod
    def _validate_args(k, alpha, m, n_replications):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("k must be a positive integer")
        if not (isinstance(n_replications, int) and n_replications > 0):
            raise ValueError("n_replications must be a positive integer")
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
        if isinstance(X, np.ndarray):
            X = X.copy()
        else:
            X = X.values.copy()
        
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
