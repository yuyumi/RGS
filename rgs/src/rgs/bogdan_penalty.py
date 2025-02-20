import numpy as np
from sklearn.metrics import make_scorer

def create_bogdan_scorer(sigma, n, p):
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
        def penalized_score(y, y_pred):
            # Compute MSE
            error = y - y_pred
            mse = (error ** 2).mean()
            
            # Compute penalty using current k value
            penalty = 2*sigma**2/n*k*np.log(p/k)
            
            # Return negative since sklearn maximizes scores
            return -(mse + penalty)
        
        # Create sklearn-compatible scorer
        return make_scorer(penalized_score)
    
    return make_k_scorer