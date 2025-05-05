import numpy as np
from sklearn.metrics import make_scorer

def create_mse_scorer(sigma, n, p):
    """
    Create a base scorer function that can be wrapped for each k value.
    
    Parameters
    ----------
    sigma2 : float
        True noise variance
    n : int
        Sample size
    p : int
        Number of features
        
    Returns
    -------
    callable
        Function that creates sklearn-compatible scorer for a given k
    """
    def make_k_scorer(k):
        """Create a scorer for a specific k value."""
        def mse_score(y, y_pred):
            # Compute MSE
            error = y - y_pred
            mse = (error ** 2).mean()
            return -mse
        
        # Create sklearn-compatible scorer
        return make_scorer(mse_score)
    
    return make_k_scorer