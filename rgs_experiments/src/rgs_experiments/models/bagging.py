import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer

class BaggedGS(BaseEstimator, RegressorMixin):
    """
    Bagged Greedy Selection with CV for optimal k.
    First creates an ensemble through bootstrap sampling,
    then finds the optimal k for the whole ensemble.
    
    Parameters
    ----------
    k_max : int
        The maximum number of features to select.
    
    n_estimators : int, default=1000
        Number of bagged estimators for the ensemble.
        
    random_state : int or None, default=None
        Random number generator seed.
        
    cv : int, default=5
        Number of cross-validation folds.
        
    scoring : string, callable, or None, default=None
        Scoring method to use. If None, defaults to 'neg_mean_squared_error'.
        If string, uses sklearn's scoring methods.
        If callable, expects a function with signature scorer(y_true, y_pred).
    """
    def __init__(self, k_max, n_estimators=1000, random_state=None, cv=5, scoring=None):
        self.k_max = k_max
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        
    def _get_scorer(self, k):
        """Get a scoring function for the current k value."""
        if self.scoring is None:
            return get_scorer('neg_mean_squared_error')
        elif isinstance(self.scoring, str):
            return get_scorer(self.scoring)
        elif callable(self.scoring):
            scorer = self.scoring(k)
            return scorer
        else:
            raise ValueError("scoring should be None, a string, or a callable")
    
    def _fit_individual_gs(self, X, y, bootstrap=True, random_state=None):
        """Fit a single GS model with k_max steps using RGS forward selection criterion."""
        n, p = X.shape
        generator = np.random.RandomState(random_state)
        
        if bootstrap:
            # Bootstrap sample indices
            indices = generator.choice(n, size=n, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Center the data
        X_centered = X_sample - X_sample.mean(axis=0)
        y_centered = y_sample - y_sample.mean()
        y_mean = y_sample.mean()
        
        # Initialize storage for all k steps
        coefs = np.zeros((self.k_max + 1, p))
        intercepts = np.full(self.k_max + 1, y_mean)
        selected_features = [[] for _ in range(self.k_max + 1)]
        
        # Pre-compute feature norms for normalization
        feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
        feature_norms[feature_norms < 1e-10] = 1.0
        
        # Initial residuals
        residuals = y_centered.copy()
        
        # Initialize QR factorization
        Q = np.empty((n, 0))
        R = np.empty((0, 0))
        
        # Track selected features efficiently
        unselected_mask = np.ones(p, dtype=bool)
        selected_indices = []
        
        for k in range(1, self.k_max + 1):
            # Check for remaining features
            if not np.any(unselected_mask):
                # No more features, copy previous values
                coefs[k] = coefs[k-1]
                intercepts[k] = intercepts[k-1]
                selected_features[k] = selected_features[k-1].copy()
                continue
                
            # Get remaining indices efficiently
            remaining_indices = np.where(unselected_mask)[0]
            
            # Calculate forward selection criterion (same as RGS)
            X_candidates = X_centered[:, remaining_indices]
            correlations = np.abs(residuals @ X_candidates)
            
            if Q.shape[1] == 0:
                # First feature selection: normalize by feature norms
                fs_values = correlations / feature_norms[remaining_indices]
            else:
                # Cache Q transpose for efficiency
                Q_T = Q.T
                
                # Compute orthogonal components
                proj_matrix = Q_T @ X_candidates
                x_orth = X_candidates - Q @ proj_matrix
                orth_norms = np.linalg.norm(x_orth, axis=0)
                
                # Handle numerical issues
                valid_mask = orth_norms > 1e-10
                fs_values = np.full(len(remaining_indices), -np.inf)
                fs_values[valid_mask] = correlations[valid_mask] / orth_norms[valid_mask]
            
            # Select best feature based on forward selection criterion
            best_idx_rel = np.argmax(fs_values)
            best_feature = remaining_indices[best_idx_rel]
            
            # Update selection tracking
            selected_indices.append(best_feature)
            unselected_mask[best_feature] = False
            selected_features[k] = selected_indices.copy()
            
            # Update QR factorization incrementally
            x_new = X_centered[:, best_feature]
            if k == 1:
                # Initialize QR
                q_norm = np.linalg.norm(x_new)
                Q = x_new.reshape(-1, 1) / q_norm
                R = np.array([[q_norm]])
            else:
                # Update QR
                r_k = Q.T @ x_new
                q_k_plus_1 = x_new - Q @ r_k
                q_k_plus_1_norm = np.linalg.norm(q_k_plus_1)
                
                if q_k_plus_1_norm > 1e-10:
                    # Normalize new Q component
                    q_k_plus_1 = q_k_plus_1 / q_k_plus_1_norm
                    
                    # Extend Q
                    Q = np.column_stack([Q, q_k_plus_1])
                    
                    # Extend R
                    R_new = np.zeros((k, k))
                    R_new[:k-1, :k-1] = R
                    R_new[:k-1, k-1] = r_k
                    R_new[k-1, k-1] = q_k_plus_1_norm
                    R = R_new
            
            # Solve for coefficients using QR factorization
            beta = np.linalg.solve(R, Q.T @ y_centered)
            
            # Update coefficients
            coefs[k, selected_indices] = beta
            
            # Update residuals only when needed
            if k < self.k_max:
                residuals = y_centered - Q @ (Q.T @ y_centered)
        
        return coefs, intercepts, selected_features
    
    def fit(self, X, y):
        """
        Fit the ensemble and find optimal k through cross-validation.
        First creates all ensemble members, then evaluates different k values.
        """
        n_samples, n_features = X.shape
        
        if self.cv == 1:
            # No CV - fit all estimators on full dataset
            all_estimators = []
            for i in range(self.n_estimators):
                seed = None if self.random_state is None else self.random_state + i
                coefs, intercepts, features = self._fit_individual_gs(
                    X, y, bootstrap=True, random_state=seed
                )
                all_estimators.append((coefs, intercepts, features))
            
            # Find optimal k by evaluating on training data
            self.cv_scores_ = {}
            for k in range(1, self.k_max + 1):
                y_pred = self._predict_with_k(X, k, all_estimators)
                scorer = self._get_scorer(k)
                score = scorer._score_func(y, y_pred)
                self.cv_scores_[k] = score
                
            # Select best k
            self.k_ = max(self.cv_scores_.items(), key=lambda x: x[1])[0]
            self.estimators_ = all_estimators
            
        else:
            # Setup CV
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, 
                            random_state=self.random_state) if isinstance(self.cv, int) else self.cv
            
            # For each fold
            self.cv_scores_ = {k: [] for k in range(1, self.k_max + 1)}
            
            for train_idx, val_idx in cv_splitter.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit all ensemble members on training data
                fold_estimators = []
                for i in range(self.n_estimators):
                    seed = None if self.random_state is None else self.random_state + i
                    coefs, intercepts, features = self._fit_individual_gs(
                        X_train, y_train, bootstrap=True, random_state=seed
                    )
                    fold_estimators.append((coefs, intercepts, features))
                
                # Evaluate each k using the whole ensemble
                for k in range(1, self.k_max + 1):
                    y_pred = self._predict_with_k(X_val, k, fold_estimators)
                    scorer = self._get_scorer(k)
                    score = scorer._score_func(y_val, y_pred)
                    self.cv_scores_[k].append(score)
            
            # Find best k based on average CV score
            mean_scores = {k: np.mean(scores) for k, scores in self.cv_scores_.items()}
            self.k_ = max(mean_scores.items(), key=lambda x: x[1])[0]
            
            # Fit final ensemble on full dataset
            self.estimators_ = []
            for i in range(self.n_estimators):
                seed = None if self.random_state is None else self.random_state + i
                coefs, intercepts, features = self._fit_individual_gs(
                    X, y, bootstrap=True, random_state=seed
                )
                self.estimators_.append((coefs, intercepts, features))
                
        return self
    
    def _predict_with_k(self, X, k, estimators):
        """Make predictions using k features for all estimators."""
        n_samples = X.shape[0]
        predictions = np.zeros((len(estimators), n_samples))
        
        for i, (coefs, intercepts, _) in enumerate(estimators):
            predictions[i] = X @ coefs[k] + intercepts[k]
            
        # Average predictions across estimators
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        """Predict using the optimal k value."""
        return self._predict_with_k(X, self.k_, self.estimators_)