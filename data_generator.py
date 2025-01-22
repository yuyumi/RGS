import numpy as np
from scipy.stats import ortho_group

def generate_orthogonal_X(n_predictors = 250, n_train=2000, seed=123):
    """
    Generate the base design matrix X that will remain fixed across replications.
    """
    np.random.seed(seed)

    X = np.random.randn(n_train, n_predictors)
    Q, R = np.linalg.qr(X)
    X = Q * np.sqrt(n_train)
    
    correlation_matrix = (X.T @ X) / n_train
    print(f"Matrix is orthogonal: {np.allclose(correlation_matrix, np.eye(n_predictors), atol=1e-10)}")
    
    off_diag_mask = ~np.eye(n_predictors, dtype=bool)
    max_correlation = np.max(np.abs(correlation_matrix[off_diag_mask]))
    print(f"Maximum absolute off-diagonal correlation: {max_correlation:.2e}")
    
    return X

def generate_banded_X(n_predictors=250, n_train=2000, seed=123):
    """
    Generate a design matrix X with AR(1)-like correlation structure.
    Each block is transformed by a different random orthogonal matrix.
    """
    np.random.seed(seed)
    gamma = 0.65
    
    # Create correlation matrix
    indices = np.arange(n_predictors)
    distances = np.abs(indices[:, np.newaxis] - indices)
    gram_matrix = gamma**distances
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
    
    if np.any(eigenvals < -1e-10):
        raise ValueError("Correlation matrix has negative eigenvalues")
    
    # Create base matrix with desired covariance
    X_base = eigenvecs @ np.diag(np.sqrt(np.abs(eigenvals))) @ eigenvecs.T
    
    n_per_batch = n_predictors
    n_full_repeats = n_train // n_per_batch
    remainder = n_train % n_per_batch
    
    # Create the repeated structure first
    X = np.zeros((n_train, n_predictors))
    
    # Fill in full blocks
    for i in range(n_full_repeats):
        Q = ortho_group.rvs(n_per_batch)
        start_idx = i * n_per_batch
        end_idx = (i + 1) * n_per_batch
        X[start_idx:end_idx] = Q @ X_base
    
    # Handle remainder if any
    if remainder > 0:
        Q = ortho_group.rvs(n_per_batch)
        start_idx = n_full_repeats * n_per_batch
        X[start_idx:] = (Q @ X_base)[:remainder]
    
    # Scale to match target covariance
    X = X * np.sqrt(n_predictors/n_train)
    
    # Verify gram matrix structure
    realized_gram = X.T @ X
    max_diff = np.max(np.abs(realized_gram - gram_matrix))
    print(f"Maximum deviation from target in gram matrix: {max_diff:.2e}")
    
    return X

def generate_exact_sparsity_example(X, signal_proportion=0.04, sigma=25, seed=123):
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_inexact_sparsity_example(X, signal_proportion=0.04, sigma=25, eta=0.2, seed=123):
    """
    Generate response variable with inexact sparsity structure controlled by eta.
    First p*signal_proportion coefficients are strong signals (magnitude 1),
    remaining coefficients decay exponentially at rate eta.
    
    Parameters:
    -----------
    X : ndarray of shape (n_train, n_features)
        Design matrix
    eta : float, default=0.1
        Controls decay of coefficient magnitudes. 
        Larger eta -> faster decay -> more sparsity
        Smaller eta -> slower decay -> less sparsity
    sigma : float, default=25
        Noise standard deviation
    seed : int, default=123
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray of shape (n_train, n_features)
        Design matrix
    y : ndarray of shape (n_train,)
        Noisy response variable
    y_true : ndarray of shape (n_train,)
        Noiseless response variable
    p : int
        Number of features
    sigma : float
        Noise standard deviation
    """
    np.random.seed(seed)
    n_train, p = X.shape
    signals = int(p*signal_proportion)
    
    # Create exponentially decaying coefficients after first few
    beta = np.zeros(p)
    beta[:signals] = 1.0  # Strong signals
    
    # Generate random signs for remaining coefficients
    signs = np.random.choice([-1, 1], size=p-signals)
    
    # Create exponentially decaying magnitudes
    indices = np.arange(p-signals)
    magnitudes = np.exp(-eta * indices)
    
    # Combine signs and magnitudes for remaining coefficients
    beta[signals:] = signs * magnitudes
    
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)
    
    return X, y, y_true, beta, p, sigma

def generate_nonlinear_example(X, signal_proportion=0.04, sigma = 10, eta=0.5, seed=123):
    """
    Generate example data with controllable nonlinearity.
    
    Parameters
    ----------
    X : ndarray
        Input design matrix
    eta : float, optional (default=0.5)
        Nonlinearity parameter between 0 and 1
        eta = 0: fully linear model
        eta = 1: highly nonlinear model
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray
        Input design matrix
    y : ndarray
        Noisy observations
    y_true : ndarray
        True signal (without noise)
    p : int
        Number of predictors
    sigma : float
        Noise level
    """
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    # Linear component (similar to other examples)
    beta_linear = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    linear_signal = X @ beta_linear
    
    # Nonlinear component using interactions and quadratic terms
    # Select first signals/2 active variables for interactions
    X_active = X[:, :signals//2]
    
    # Create quadratic terms
    quad_terms = X_active**2
    
    # Create pairwise interactions
    interactions = np.zeros(n_train)
    for i in range(4):
        for j in range(i+1, 5):
            interactions += X_active[:, i] * X_active[:, j]
    
    # Combine linear and nonlinear components
    nonlinear_signal = (quad_terms.sum(axis=1) + interactions) / np.sqrt(p)
    
    # Mix linear and nonlinear signals based on eta
    y_true = (1 - eta) * linear_signal + eta * nonlinear_signal
    
    # Add noise
    y = y_true + np.random.normal(0, sigma, n_train)
    
    return X, y, y_true, beta_linear, p, sigma

def generate_laplace_example(X, signal_proportion=0.04, sigma = 15, seed=123):
    """
    Generate example data with Laplace noise.
    The Laplace distribution has similar mean but heavier tails than Gaussian.
    
    Parameters
    ----------
    X : ndarray
        Input design matrix
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray
        Input design matrix
    y : ndarray
        Noisy observations
    y_true : ndarray
        True signal (without noise)
    p : int
        Number of predictors
    sigma : float
        Scale parameter of Laplace distribution
    """
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    # Use same beta structure as other examples
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    
    # Generate true signal
    y_true = X @ beta
    
    # Generate Laplace noise
    # Use scale = sigma/sqrt(2) to match the variance of N(0,sigma^2)
    laplace_noise = np.random.laplace(loc=0, scale=sigma/np.sqrt(2), size=n_train)
    
    # Add noise to signal
    y = y_true + laplace_noise
    
    return X, y, y_true, beta, p, sigma

def generate_cauchy_example(X, signal_proportion=0.04, scale=15, seed=123):
    """
    Generate example data with Cauchy noise.
    The Cauchy distribution has undefined moments (infinite mean and variance)
    and extremely heavy tails, making it a challenging noise model.
    
    Parameters
    ----------
    X : ndarray
        Input design matrix
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray
        Input design matrix
    y : ndarray
        Noisy observations
    y_true : ndarray
        True signal (without noise)
    p : int
        Number of predictors
    scale : float
        Scale parameter of Cauchy distribution (analogous to σ but not equivalent)
    """
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    # Use same beta structure as other examples
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    
    # Generate true signal
    y_true = X @ beta
    
    # Generate Cauchy noise
    # Use standard Cauchy distribution (location=0, scale=scale)
    cauchy_noise = scale * np.random.standard_cauchy(size=n_train)
    
    # Add noise to signal
    y = y_true + cauchy_noise
    
    return X, y, y_true, beta, p, scale

