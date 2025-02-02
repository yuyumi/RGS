import numpy as np
from scipy.stats import ortho_group
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Callable, Any, Union

def make_model(params: Dict[str, Any]) -> BaseEstimator:
    """
    Creates a scikit-learn estimator object based on provided parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary containing model parameters
        
    Returns
    -------
    BaseEstimator
        A scikit-learn estimator object
    """
    model_type = params.get('model_type')
    if model_type == 'rgs':
        from rgs import RGS
        return RGS(**params['model_params'])
    # Add more model types as needed
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def make_dgp(params: Dict[str, Any]) -> Tuple[Callable, Callable, Callable]:
    """
    Creates data generating process functions based on provided parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary containing DGP parameters including:
        - design: str, type of design matrix ('orthogonal', 'banded', 'block')
        - n_train: int, number of training samples
        - n_predictors: int, number of features
        - signal_type: str, type of signal ('exact', 'inexact', 'nonlinear', 'laplace', 'cauchy')
        - signal_proportion: float, proportion of active features
        - sigma: float, noise level
        - block_params: dict, optional parameters for block design
        - eta: float, optional parameter for inexact/nonlinear cases
        
    Returns
    -------
    tuple
        (make_X, make_y_train, make_y_test) functions
    """
    design = params.get('design', 'orthogonal')
    n_train = params.get('n_train', 2000)
    n_predictors = params.get('n_predictors', 250)
    signal_type = params.get('signal_type', 'exact')
    signal_proportion = params.get('signal_proportion', 0.04)
    sigma = params.get('sigma', 25)
    seed = params.get('seed', 123)
    eta = params.get('eta', 0.2)
    
    def make_X() -> np.ndarray:
        """Generate covariate data."""
        if design == 'orthogonal':
            return generate_orthogonal_X(n_predictors, n_train, seed)
        elif design == 'banded':
            return generate_banded_X(n_predictors, n_train, seed)
        elif design == 'block':
            block_size = params.get('block_params', {}).get('block_size', 50)
            within_correlation = params.get('block_params', {}).get('within_correlation', 0.7)
            return generate_block_X(n_predictors, n_train, block_size, within_correlation, seed)
        else:
            raise ValueError(f"Unknown design type: {design}")
    
    def make_y_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training response data."""
        if signal_type == 'exact':
            _, y, y_true, beta, _, _ = generate_exact_sparsity_example(X, signal_proportion, sigma, seed)
        elif signal_type == 'inexact':
            _, y, y_true, beta, _, _ = generate_inexact_sparsity_example(X, signal_proportion, sigma, eta, seed)
        elif signal_type == 'nonlinear':
            _, y, y_true, beta, _, _ = generate_nonlinear_example(X, signal_proportion, sigma, eta, seed)
        elif signal_type == 'laplace':
            _, y, y_true, beta, _, _ = generate_laplace_example(X, signal_proportion, sigma, seed)
        elif signal_type == 'cauchy':
            _, y, y_true, beta, _, _ = generate_cauchy_example(X, signal_proportion, sigma, seed)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        return y, beta
    
    def make_y_test(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Generate test response data using same beta."""
        y_true = X @ beta
        if signal_type == 'laplace':
            noise = np.random.laplace(0, sigma/np.sqrt(2), size=len(X))
        elif signal_type == 'cauchy':
            noise = sigma * np.random.standard_cauchy(size=len(X))
        else:
            noise = np.random.normal(0, sigma, size=len(X))
        return y_true + noise
    
    return make_X, make_y_train, make_y_test

def get_model_score(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = 'mse'
) -> Union[float, Dict[str, float]]:
    """
    Compute model performance metrics.
    
    Parameters
    ----------
    model : BaseEstimator
        Fitted model
    X_train : ndarray
        Training features
    y_train : ndarray
        Training response
    X_test : ndarray
        Test features
    y_test : ndarray
        Test response
    metric : str, default='mse'
        Metric to compute ('mse', 'r2', or 'all')
        
    Returns
    -------
    float or dict
        Score(s) for the model
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if metric == 'mse':
        return mean_squared_error(y_test, y_test_pred)
    elif metric == 'r2':
        return r2_score(y_test, y_test_pred)
    elif metric == 'all':
        return {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Import DGP functions
from sim_util_dgs import (
    generate_orthogonal_X,
    generate_banded_X,
    generate_block_X,
    generate_exact_sparsity_example,
    generate_inexact_sparsity_example,
    generate_nonlinear_example,
    generate_laplace_example,
    generate_cauchy_example
)