import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer

class SmearedGS(BaseEstimator, RegressorMixin):
    """
    Data Smearing with Greedy Selection and CV for optimal k and noise scale.
    First creates ensemble by fitting on noise-perturbed targets,
    then finds optimal k and noise scale for the whole ensemble.
    
    Parameters
    ----------
    k_max : int
        The maximum number of features to select.
    
    n_estimators : int, default=1000
        Number of estimators in the ensemble.
        
    noise_scale : float or list of float, default=1.0
        Scale of Gaussian noise to add. If list, CV selects the best scale.
        
    random_state : int or None, default=None
        Random number generator seed.
        
    cv : int, default=5
        Number of cross-validation folds.
        
    scoring : string, callable, or None, default=None
        Scoring method to use. If None, defaults to 'neg_mean_squared_error'.
        If string, uses sklearn's scoring methods.
        If callable, expects a function with signature scorer(y_true, y_pred).
    """
    def __init__(self, k_max, n_estimators=1000, noise_scale=1.0, 
                 random_state=None, cv=5, scoring=None):
        self.k_max = k_max
        self.n_estimators = n_estimators
        self.noise_scale = noise_scale
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
    
    def _fit_individual_gs(self, X, y, noise_scale, random_state=None):
        """Fit a single GS model with k_max steps using smeared target."""
        n, p = X.shape
        generator = np.random.RandomState(random_state)
        
        # Add Gaussian noise to target
        noise = generator.normal(0, noise_scale, size=len(y))
        y_smeared = y + noise
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        y_centered = y_smeared - y_smeared.mean()
        y_mean = y_smeared.mean()
        
        # Initialize storage for all k steps
        coefs = np.zeros((self.k_max + 1, p))
        intercepts = np.zeros(self.k_max + 1)
        selected_features = [[] for _ in range(self.k_max + 1)]
        
        # For k=0, empty model
        intercepts[0] = y_mean
        
        # Initial residuals are centered y
        residuals = y_centered.copy()
        
        # Scale X for correlations
        X_norms = np.sqrt(np.sum(X_centered ** 2, axis=0) + 1e-10)
        X_scaled = X_centered / X_norms
        
        # Greedy forward selection
        selected_set = set()
        
        for k in range(1, self.k_max + 1):
            # Get remaining features
            remaining = list(set(range(p)) - selected_set)
            if not remaining:
                # If we've selected all features, just copy the previous step
                coefs[k] = coefs[k-1]
                intercepts[k] = intercepts[k-1]
                selected_features[k] = selected_features[k-1].copy()
                continue
                
            # Calculate correlations with current residuals
            correlations = np.abs(X_scaled[:, remaining].T @ residuals)
            
            # Select best feature
            best_idx_pos = np.argmax(correlations)
            best_feature = remaining[best_idx_pos]
            selected_set.add(best_feature)
            selected_features[k] = list(selected_set)
            
            # Fit least squares on selected features
            features = selected_features[k]
            X_selected = X_centered[:, features]
            beta, *_ = np.linalg.lstsq(X_selected, y_centered, rcond=None)
            
            # Update coefficients and intercept
            coefs[k, features] = beta
            intercepts[k] = y_mean
            
            # Update residuals
            residuals = y_centered - X_selected @ beta
        
        return coefs, intercepts, selected_features
    
    def fit(self, X, y):
        """
        Fit the ensemble and find optimal k and noise_scale through cross-validation.
        """
        # Convert noise_scale to list if it's a single value
        noise_scales = self.noise_scale if isinstance(self.noise_scale, list) else [self.noise_scale]
        
        if self.cv == 1:
            # No CV - fit on full dataset and evaluate all combinations
            estimator_sets = {scale: [] for scale in noise_scales}
            
            # Fit all ensemble members for each noise scale
            for scale in noise_scales:
                for i in range(self.n_estimators):
                    seed = None if self.random_state is None else self.random_state + i
                    coefs, intercepts, features = self._fit_individual_gs(
                        X, y, scale, random_state=seed
                    )
                    estimator_sets[scale].append((coefs, intercepts, features))
            
            # Evaluate all combinations of k and noise_scale
            self.cv_scores_ = {}
            for scale in noise_scales:
                for k in range(1, self.k_max + 1):
                    y_pred = self._predict_with_k(X, k, estimator_sets[scale])
                    scorer = self._get_scorer(k)
                    score = scorer._score_func(y, y_pred)
                    self.cv_scores_[(k, scale)] = score
            
            # Find best parameters
            best_params = max(self.cv_scores_.items(), key=lambda x: x[1])[0]
            self.k_, self.noise_scale_ = best_params
            self.estimators_ = estimator_sets[self.noise_scale_]
            
        else:
            # Setup CV
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, 
                           random_state=self.random_state) if isinstance(self.cv, int) else self.cv
            
            # Initialize scores dictionary
            self.cv_scores_ = {(k, scale): [] for k in range(1, self.k_max + 1) 
                             for scale in noise_scales}
            
            # For each CV fold
            for train_idx, val_idx in cv_splitter.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # For each noise scale
                fold_estimators = {scale: [] for scale in noise_scales}
                
                for scale in noise_scales:
                    # Fit ensemble on training data
                    for i in range(self.n_estimators):
                        seed = None if self.random_state is None else self.random_state + i
                        coefs, intercepts, features = self._fit_individual_gs(
                            X_train, y_train, scale, random_state=seed
                        )
                        fold_estimators[scale].append((coefs, intercepts, features))
                
                    # Evaluate each k using the ensemble with this noise scale
                    for k in range(1, self.k_max + 1):
                        y_pred = self._predict_with_k(X_val, k, fold_estimators[scale])
                        scorer = self._get_scorer(k)
                        score = scorer._score_func(y_val, y_pred)
                        self.cv_scores_[(k, scale)].append(score)
            
            # Find best parameters based on average CV score
            mean_scores = {params: np.mean(scores) 
                         for params, scores in self.cv_scores_.items()}
            best_params = max(mean_scores.items(), key=lambda x: x[1])[0]
            self.k_, self.noise_scale_ = best_params
            
            # Fit final ensemble on full dataset with best noise scale
            self.estimators_ = []
            for i in range(self.n_estimators):
                seed = None if self.random_state is None else self.random_state + i
                coefs, intercepts, features = self._fit_individual_gs(
                    X, y, self.noise_scale_, random_state=seed
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