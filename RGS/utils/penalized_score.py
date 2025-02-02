import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, make_scorer

def create_penalized_scorer(sigma2, n, p, k):
    """
    Create a scorer function for penalized MSE that's compatible with sklearn.
    
    Parameters
    ----------
    sigma2 : float
        True noise variance
    n : int
        Sample size
    p : int
        Number of features
    k : int
        Number of features to use
        
    Returns
    -------
    callable
        Scorer function compatible with sklearn
    """
    def penalized_score(y_true, y_pred):
        # Compute MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # Compute penalty
        penalty = 2*sigma2**2/n*k*np.log(p/k)
        
        # Return negative since sklearn maximizes scores
        return -(mse + penalty)
    
    # Create sklearn-compatible scorer
    return make_scorer(penalized_score)
