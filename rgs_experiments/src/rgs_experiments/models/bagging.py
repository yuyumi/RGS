import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from rgs import RGS  # Import RGS class

class BaggedGS(BaseEstimator, RegressorMixin):
    """
    Bagged Greedy Selection with CV for optimal k.
    Uses RGS with m=p and B=1 for each bootstrap sample.
    
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
    def __init__(self, k_max, n_estimators=500, random_state=None, cv=5, scoring=None):
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
        """Fit a single GS model using RGS with m=p and B=1."""
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
        
        # Use RGS with m=p (all features) and B=1 (one replicate)
        rgs = RGS(
            k_max=self.k_max,
            m=p,  # Use all features as candidates
            n_estimators=1,  # Single replicate
            random_state=random_state
        )
        rgs.fit(X_sample, y_sample)
        
        # Extract coefficients and intercepts for all k values
        coefs = np.array([coef for coef in rgs.coef_])
        intercepts = np.array([intercept for intercept in rgs.intercept_])
        
        # Extract selected features
        selected_features = [[] for _ in range(self.k_max + 1)]
        for k in range(1, self.k_max + 1):
            # Get indices of non-zero coefficients
            selected = np.where(np.abs(coefs[k]) > 1e-10)[0]
            selected_features[k] = list(selected)
        
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