import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class DataSmearingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, noise_sigma=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.noise_sigma = noise_sigma
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Create perturbed y with Gaussian noise
            noise = rng.normal(0, self.noise_sigma, size=len(y))
            y_perturbed = y + noise
            
            # Fit model on perturbed data
            estimator = LinearRegression()
            estimator.fit(X, y_perturbed)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        # Average predictions across all models
        predictions = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(predictions, axis=0)