from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, check_X_y
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class FastRandomizedGreedySelectionCV(BaseEstimator, RegressorMixin):
    """
    FastRandomizedGreedySelection is a feature selection algorithm that selects a subset of features
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


    def __init__(self, k_max, m_grid, n_estimators=1000, n_resample_iter=0, random_state=None, cv=None):
        # FastRandomizedGreedySelectionCV._validate_args(k_max, alpha, m, n_estimators)
        self.k_max = k_max
        self.m_grid = m_grid
        self.n_estimators = n_estimators
        self.n_resample_iter = n_resample_iter
        self.random_state = random_state
        if cv is None:
            self.cv = KFold(n_splits=5, random_state=self.random_state)
        else:
            self.cv = cv
        self.cv_scores_ = {k : {m : [] for m in m_grid} for k in range(1, k_max+1)}

    def fit(self, X, y):
        # print("Starting cross-validation...")
        for fold, (train_ids, val_ids) in enumerate(self.cv.split(X), 1):
            # print(f"\nFold {fold}/{self.cv.n_splits}")
            X_train = X[train_ids]
            X_val = X[val_ids]
            y_train = y[train_ids]
            y_val = y[val_ids]
            self.fit_fold(X_train, y_train, X_val, y_val)
        
        # Find best hyperparameters
        # print("\nFinding best hyperparameters...")
        best_m = np.array([max(self.cv_scores_[k], key=lambda m: np.mean(self.cv_scores_[k][m])) 
                        for k in range(1, self.k_max+1)])
        
        mean_scores = [np.mean(self.cv_scores_[k][best_m[k-1]]) 
                    for k in range(1, self.k_max+1)]
        
        # print("\nMean scores for each k:")
        # for k in range(1, self.k_max+1):
        #     print(f"k={k}: best_m={best_m[k-1]}, score={mean_scores[k-1]:.6f}")
        
        self.k_ = np.argmin(mean_scores) + 1
        self.m_ = best_m[self.k_-1]
        
        # print(f"\nSelected parameters: k={self.k_}, m={self.m_}")
        return self

    def fit_fold(self, X_train, y_train, X_val, y_val):
        # print("\nStarting new fold...")
        # Initialize
        X_train, y_train = self._validate_training_inputs(X_train, y_train)
        X_train = X_train.T  
        X_val = X_val.T
        X_centered = X_train - X_train.mean(axis=1, keepdims=True)
        X_scaled = X_centered / np.sqrt(np.sum((X_train ** 2), axis=1, keepdims=True))
        self.p = X_train.shape[0]
        generator = np.random.default_rng(self.random_state)
        
        # print(f"Initialized with {self.p} features")
        # print(f"m_grid: {self.m_grid}")
        
        # Initialize feature sets for all k
        self.feature_sets = [{} for _ in range(self.k_max + 1)]
        
        # Initialize k=0 with empty set
        for m in self.m_grid:
            self.feature_sets[0][m] = Counter({frozenset(): self.n_estimators})
        
        # print("\nInitializing k=1 based on correlations...")
        # Initialize k=1 
        residuals = y_train - y_train.mean()
        correlations = np.abs(X_scaled @ residuals)
        
        # Initialize k=1 based on correlations
        for m in self.m_grid:
            feature_sets = {}
            n_candidates = min(m, self.p)
            candidates = generator.choice(range(self.p), size=n_candidates, replace=False)
            psi = candidates[np.argmax(correlations[candidates])]
            feature_sets[frozenset({psi})] = self.n_estimators
            self.feature_sets[1][m] = Counter(feature_sets)
            # print(f"  m={m}: selected feature {psi}")
        
        self.coef_ = {m : [] for m in self.m_grid}
        self.intercept_ = {m : [] for m in self.m_grid}

        # Main loop
        for k in range(1, self.k_max + 1):
            # print(f"\nProcessing k={k}...")
            feature_set_freqs = defaultdict(dict)
            for m in self.m_grid:
                Ms = list(self.feature_sets[k][m].keys())
                freqs = list(self.feature_sets[k][m].values())
                # print(f"  m={m}: current feature sets: {[list(M) for M in Ms]}")
                
                if len(Ms) > 0:
                    Ms = [np.array(list(M), dtype=int) if len(M) > 0 else np.array([], dtype=int) for M in Ms]
                    freqs = np.array(freqs)
                    
                    if self.n_resample_iter > 0:
                        # print(f"    Resampling with {self.n_resample_iter} iterations")
                        proportions = freqs / self.n_estimators
                        freqs = generator.multinomial(self.n_estimators, proportions)
                    Ms = [M for M, f in zip(Ms, freqs) if f > 0]
                    freqs = freqs[freqs > 0]
                    
                    for M, freq in zip(Ms, freqs):
                        feature_set_freqs[tuple(M)][m] = freq
                        
            coef_ = {m : np.zeros(self.p) for m in self.m_grid}  # Moved this initialization up
            
            # Process each feature set
            for M_tuple, freqs_dict in feature_set_freqs.items():
                M = np.array(M_tuple)
                # print(f"  Processing feature set: {M}")
                
                if len(M) == 0:
                    beta = np.array([])
                    residuals = y_train - y_train.mean()
                else:
                    selected_features = X_centered[M, :]
                    beta, _, _, _ = lstsq(selected_features.T, y_train)
                    residuals = y_train - selected_features.T @ beta
                
                # Update coefficients
                for m, freq in freqs_dict.items():
                    if len(M) > 0:
                        coef_[m][M] += beta * freq
                        
                if k < self.k_max:
                    # Generate new feature sets
                    mask = np.ones(self.p, dtype=bool)
                    if len(M) > 0:
                        mask[M] = False
                    M_comp = np.arange(self.p)[mask]
                    correlations = np.abs(X_scaled[M_comp, :] @ residuals)
                    
                    new_feature_sets = self._get_new_feature_sets(
                        frozenset(M_tuple), M_comp, correlations, freqs_dict, generator
                    )
                    
                    # Print new feature sets
                    for m in self.m_grid:
                        if m in new_feature_sets:
                    #         print(f"    m={m}: new feature sets: {[list(M) for M in new_feature_sets[m].keys()]}")
                            self.feature_sets[k+1][m] = new_feature_sets[m]
            
            # Update parameters and compute CV scores
            # print(f"\n  Computing CV scores for k={k}:")
            for m in self.m_grid:
                self.coef_[m].append(coef_[m] / self.n_estimators)
                self.intercept_[m].append(
                    y_train.mean() - np.dot(X_train.mean(axis=1), coef_[m] / self.n_estimators)
                )
                
                y_preds = (X_val.T @ self.coef_[m][k-1]) + self.intercept_[m][k-1]
                score = mean_squared_error(y_val, y_preds)
                self.cv_scores_[k][m].append(score)
                # print(f"    m={m}: MSE = {score:.6f}")

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.p
        return X @ self.coef_[self.m_][self.k_] + self.intercept_[self.m_][self.k_]
        
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
        # Remove the m/alpha validation since we're using m_grid now
        assert all(m <= p for m in self.m_grid)  # Verify all m values are valid
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X, y

    def _get_new_feature_sets(self, M, M_comp, correlations, freqs_dict, generator):
        # Generate candidates for next step for each iteration
        feature_sets = {}
        for m, freq in freqs_dict.items():
            n_candidates = min(m, len(M_comp))
            candidates = np.zeros((freq, n_candidates), dtype=int)
            for iter in range(freq):
                candidates[iter, :] = generator.choice(range(len(M_comp)), size=n_candidates, replace=False)
            # Compute the top candidate feature across each iteration
            candidate_correlations = correlations[candidates.flatten()].reshape(freq, n_candidates)
            max_index_in_subset = np.argmax(candidate_correlations, axis=1)
            psi_vals = M_comp[candidates[range(freq), max_index_in_subset]]
            # Summarize the results in a Counter object
            psi_freqs = np.bincount(psi_vals)
            psi_vals_unique = np.nonzero(psi_freqs)[0]
            M_new_unique = [frozenset(set(M) | {feat}) for feat in psi_vals_unique]
            M_new_freqs = Counter(dict(zip(M_new_unique, psi_freqs[psi_vals_unique])))
            feature_sets[m] = M_new_freqs
        return feature_sets