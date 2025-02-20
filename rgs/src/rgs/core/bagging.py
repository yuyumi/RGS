import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler

class BaggedRGSCV(BaseEstimator, RegressorMixin):
    """
    Cross-validation wrapper for Bagged RGS with custom scoring support.
    
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
    
    def _fit_individual_rgs(self, X, y, random_state):
        """Fit a single RGS model with k_max steps."""
        n = X.shape[0]
        generator = np.random.RandomState(random_state)
        # Bootstrap sample indices
        indices = generator.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Center the data
        X_centered = X_boot - X_boot.mean(axis=0)
        y_centered = y_boot - y_boot.mean()
        
        # Initialize storage for all k steps
        selected_features = []
        coef_ = []
        intercept_ = []
        
        # Initial residuals are centered y
        residuals = y_centered.copy()
        
        # Scale X for correlations
        X_scaled = X_centered / np.sqrt(np.sum(X_centered ** 2, axis=0))
        
        # Greedy forward selection for k_max steps
        available_features = set(range(X.shape[1]))
        selected_features_set = set()
        
        for k in range(self.k_max + 1):
            if k == 0:
                # For k=0, empty model
                coef_.append(np.zeros(X.shape[1]))
                intercept_.append(y_boot.mean())
                selected_features.append([])
                continue
                
            # Compute correlations with residuals for remaining features
            remaining_features = list(available_features - selected_features_set)
            correlations = np.abs(X_scaled[:, remaining_features].T @ residuals)
            
            # Select feature with highest correlation
            best_idx = remaining_features[np.argmax(correlations)]
            selected_features_set.add(best_idx)
            current_features = list(selected_features_set)
            selected_features.append(current_features)
            
            # Fit least squares on selected features
            X_selected = X_centered[:, current_features]
            beta = np.linalg.lstsq(X_selected, y_centered, rcond=None)[0]
            
            # Update model coefficients
            coef_k = np.zeros(X.shape[1])
            coef_k[current_features] = beta
            coef_.append(coef_k)
            intercept_.append(y_boot.mean())
            
            # Update residuals
            residuals = y_centered - X_selected @ beta
            
        return coef_, intercept_, selected_features
        
    def fit(self, X, y):
        """Fit the bagged ensemble using cross-validation to select the best k."""
        # Initialize scores dictionary
        self.cv_scores_ = {k: [] for k in range(1, self.k_max + 1)}
        
        if self.cv == 1:
            # No CV - use full dataset directly
            self.estimators_ = []
            for i in range(self.n_estimators):
                coef_, intercept_, features = self._fit_individual_rgs(
                    X, y, self.random_state + i if self.random_state else None
                )
                self.estimators_.append((coef_, intercept_, features))
            
            # Evaluate each k
            for k in range(1, self.k_max + 1):
                y_pred = self._predict_k(X, k)
                scorer = self._get_scorer(k)
                score = scorer._score_func(y, y_pred)
                self.cv_scores_[k] = [score]
                
        else:
            # Setup CV splitter
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, 
                            random_state=self.random_state) if isinstance(self.cv, int) else self.cv
            
            # Perform CV
            for train_idx, val_idx in cv_splitter.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit ensemble on training data
                fold_estimators = []
                for i in range(self.n_estimators):
                    coef_, intercept_, features = self._fit_individual_rgs(
                        X_train, y_train, 
                        self.random_state + i if self.random_state else None
                    )
                    fold_estimators.append((coef_, intercept_, features))
                
                # Evaluate each k using the full ensemble
                for k in range(1, self.k_max + 1):
                    y_pred = self._predict_k_with_estimators(X_val, k, fold_estimators)
                    scorer = self._get_scorer(k)
                    score = scorer._score_func(y_val, y_pred)
                    self.cv_scores_[k].append(score)
        
        # Find optimal k
        mean_scores = {k: np.mean(scores) for k, scores in self.cv_scores_.items()}
        self.k_ = max(mean_scores.items(), key=lambda x: x[1])[0]
        
        # Fit final ensemble with best k
        self.estimators_ = []
        for i in range(self.n_estimators):
            coef_, intercept_, features = self._fit_individual_rgs(
                X, y, self.random_state + i if self.random_state else None
            )
            self.estimators_.append((coef_, intercept_, features))
        
        return self
    
    def _predict_k_with_estimators(self, X, k, estimators):
        """Make predictions for a specific k using given estimators."""
        predictions = np.zeros((len(estimators), X.shape[0]))
        for i, (coef_, intercept_, _) in enumerate(estimators):
            predictions[i] = X @ coef_[k] + intercept_[k]
        return np.mean(predictions, axis=0)
    
    def _predict_k(self, X, k):
        """Make predictions for a specific k using the fitted ensemble."""
        return self._predict_k_with_estimators(X, k, self.estimators_)
    
    def predict(self, X):
        """Make predictions using the fitted ensemble with best k."""
        return self._predict_k(X, self.k_)