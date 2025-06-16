import numpy as np
from scipy.stats import ortho_group

__all__ = [
    'generate_banded_X',
    'generate_block_X',
    'generate_exact_sparsity_example',
    'generate_inexact_sparsity_example',
    'generate_nonlinear_example',
    'generate_laplace_example',
    'generate_cauchy_example',
    'generate_spaced_sparsity_example'
]

# Helper functions to reduce redundancy

def _validate_sigma(sigma):
    """Validate sigma parameter."""
    if sigma is None:
        raise ValueError("sigma parameter must be provided")

def _create_fixed_design_matrix(target_covariance, n_train, n_predictors):
    """
    Create fixed design matrix from target covariance.
    
    Returns:
    --------
    X : ndarray
        Design matrix such that X^T X / n ≈ target_covariance
    """
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(target_covariance)
    
    if np.any(eigenvals < -1e-10):
        raise ValueError(f"Correlation matrix has negative eigenvalues: {min(eigenvals):.2e}")
    
    # Create base matrix with desired covariance
    X_base = eigenvecs @ np.diag(np.sqrt(np.abs(eigenvals))) @ eigenvecs.T
    
    n_per_batch = n_predictors
    n_full_repeats = n_train // n_per_batch
    remainder = n_train % n_per_batch
    
    # Create the repeated structure
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
    
    X = X - X.mean(axis=0)
    
    # Scale to match target covariance
    X = X * np.sqrt(n_predictors)
    
    return X



def generate_banded_X(n_predictors, n_train, rho=0.65, seed=123, fixed_design=False):
    """
    Generate a design matrix X with AR(1)-like correlation structure.
    
    Parameters:
    -----------
    n_predictors : int
        Number of predictor variables/columns in X
    n_train : int
        Number of samples/rows in X
    rho : float, default=0.65
        Correlation decay parameter for AR(1) structure
    seed : int, default=123
        Random seed for reproducibility
    fixed_design : bool, default=False
        If True: construct X such that X^T X / n = target_covariance (fixed design)
        If False: draw each row from N(0, target_covariance) (random design)
        
    Returns:
    --------
    X : ndarray of shape (n_train, n_predictors)
        Design matrix with AR(1) correlation structure
    gram_matrix : ndarray of shape (n_predictors, n_predictors)
        The target correlation matrix
    """
    np.random.seed(seed)
    
    # Create correlation matrix
    indices = np.arange(n_predictors)
    distances = np.abs(indices[:, np.newaxis] - indices)
    gram_matrix = rho**distances
    
    if fixed_design:
        # Fixed design: construct X such that X^T X / n = target_covariance
        X = _create_fixed_design_matrix(gram_matrix, n_train, n_predictors)
    else:
        # Random design: draw each row from N(0, target_covariance)
        X = np.random.multivariate_normal(
            mean=np.zeros(n_predictors),
            cov=gram_matrix,
            size=n_train
        )

    # Verify centering and covariance structure
    print(f"Design type: {'Fixed' if fixed_design else 'Random'}")
    print(f"Max column mean: {np.max(np.abs(X.mean(axis=0))):.2e}")
    
    X_centered = X - X.mean(axis=0)
    realized_cov = X_centered.T @ X_centered / (n_train)
    max_diff = np.max(np.abs(realized_cov - gram_matrix))
    print(f"Maximum deviation from target in covariance matrix: {max_diff:.2e}")
    
    return X, gram_matrix

def generate_block_X(n_predictors, n_train, block_size, within_correlation=0.7, seed=123, fixed_design=False):
    """
    Generate a design matrix X with block correlation structure where variables 
    are grouped by modulo assignment to create evenly distributed blocks.
    
    Parameters:
    -----------
    n_predictors : int
        Number of predictor variables/columns in X
    n_train : int
        Number of samples/rows in X
    block_size : int
        Number of variables in each block. n_predictors must be divisible by block_size
    within_correlation : float, default=0.7
        Correlation between variables within the same block
    seed : int, default=123
        Random seed for reproducibility
    fixed_design : bool, default=False
        If True: construct X such that X^T X / n = target_covariance (fixed design)
        If False: draw each row from N(0, target_covariance) (random design)
        
    Returns:
    --------
    X : ndarray of shape (n_train, n_predictors)
        Design matrix with the specified block correlation structure
    target_covariance : ndarray of shape (n_predictors, n_predictors)
        The target covariance matrix
    """
    
    np.random.seed(seed)
    
    # Verify parameters
    if n_predictors % block_size != 0:
        raise ValueError("n_predictors must be divisible by block_size")
    
    # Calculate number of blocks
    num_blocks = n_predictors // block_size
    
    # Create target covariance matrix
    target_covariance = np.zeros((n_predictors, n_predictors))
    
    # Fill with block structure - ones on diagonal and within_correlation for same block
    # Use modulo assignment: variables with same (i % num_blocks) are in same block
    for i in range(n_predictors):
        for j in range(n_predictors):
            if i == j:  # Diagonal
                target_covariance[i, j] = 1.0
            elif i % num_blocks == j % num_blocks:  # Same block (modulo grouping)
                target_covariance[i, j] = within_correlation
    
    if fixed_design:
        # Fixed design: construct X such that X^T X / n = target_covariance
        X = _create_fixed_design_matrix(target_covariance, n_train, n_predictors)
    else:
        # Random design: draw each row from N(0, target_covariance)
        X = np.random.multivariate_normal(
            mean=np.zeros(n_predictors),
            cov=target_covariance,
            size=n_train
        )

    # Verify centering and covariance structure
    print(f"Design type: {'Fixed' if fixed_design else 'Random'}")
    print(f"Max column mean: {np.max(np.abs(X.mean(axis=0))):.2e}")
    
    # Compute realized covariance
    X_centered = X - X.mean(axis=0)
    realized_cov = X_centered.T @ X_centered / (n_train)
    max_diff = np.max(np.abs(realized_cov - target_covariance))
    print(f"Maximum deviation from target in covariance matrix: {max_diff:.2e}")
    
    # Verify block structure
    for b in range(min(5, num_blocks)):  # Show just first 5 blocks
        block_cols = [i for i in range(n_predictors) if i % num_blocks == b]
        
        # Extract correlation sub-matrix for this block
        block_realized = realized_cov[np.ix_(block_cols, block_cols)]
        
        # Calculate mean off-diagonal correlation
        off_diag = block_realized.copy()
        np.fill_diagonal(off_diag, 0)  # Zero out diagonal
        avg_corr = np.sum(off_diag) / (len(block_cols) * (len(block_cols) - 1)) if len(block_cols) > 1 else 0
        
        print(f"Block {b} (vars {block_cols}) - Mean correlation: {avg_corr:.6f} (target: {within_correlation:.6f})")
    
    if num_blocks > 5:
        print(f"... and {num_blocks - 5} more blocks with similar structure")
    
    # Find maximum between-block correlation
    max_between = 0
    max_between_idx = None
    for i in range(n_predictors):
        for j in range(i+1, n_predictors):
            if i % num_blocks != j % num_blocks:  # Different blocks (modulo grouping)
                if abs(realized_cov[i, j]) > max_between:
                    max_between = abs(realized_cov[i, j])
                    max_between_idx = (i, j)
                    
    if max_between_idx:
        print(f"Maximum between-block correlation: {max_between:.6e} at indices {max_between_idx}")
    
    return X, target_covariance

def generate_exact_sparsity_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with exact sparsity."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    from .snr_utils import _construct_beta_vector
    beta = _construct_beta_vector(p, signal_proportion, 'exact', seed=seed)
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_spaced_sparsity_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with evenly spaced signals across all predictors."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    from .snr_utils import _construct_beta_vector
    beta = _construct_beta_vector(p, signal_proportion, 'spaced', seed=seed)
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_inexact_sparsity_example(X, signal_proportion, sigma=None, eta=0.5, seed=123):
    """Generate example with inexact sparsity."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    from .snr_utils import _construct_beta_vector
    beta = _construct_beta_vector(p, signal_proportion, 'inexact', eta=eta, seed=seed)
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)
    
    return X, y, y_true, beta, p, sigma

def generate_nonlinear_example(X, signal_proportion, sigma=None, eta=0.5, seed=123):
    """Generate example with nonlinear components."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    signals = int(p * signal_proportion)
    s2 = signals // 2  # Number of nonlinear variables
    
    # Linear component
    from .snr_utils import _construct_beta_vector
    beta_linear = _construct_beta_vector(p, signal_proportion, 'nonlinear', seed=seed)
    linear_signal = X @ beta_linear
    
    # Nonlinear component using interactions and quadratic terms
    if s2 > 0:
        X_active = X[:, :s2]
        quad_terms = X_active**2
        
        # Create pairwise interactions - only if we have at least 2 nonlinear variables
        interactions = np.zeros(n_train)
        if s2 >= 2:  # Need at least 2 variables for interactions
            for i in range(s2):
                for j in range(i+1, s2):
                    interactions += X_active[:, i] * X_active[:, j]
        
        # Combine linear and nonlinear components
        nonlinear_signal = (quad_terms.sum(axis=1) + interactions) / np.sqrt(p)
    else:
        # No nonlinear variables (s2 = 0)
        nonlinear_signal = np.zeros(n_train)
    
    y_true = (1 - eta) * linear_signal + eta * nonlinear_signal
    y = y_true + np.random.normal(0, sigma, n_train)
    
    return X, y, y_true, beta_linear, p, sigma

def generate_laplace_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with Laplace noise."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    from .snr_utils import _construct_beta_vector
    beta = _construct_beta_vector(p, signal_proportion, 'laplace', seed=seed)
    y_true = X @ beta
    
    # Generate Laplace noise (scale = sigma/sqrt(2) to match N(0,sigma^2) variance)
    laplace_noise = np.random.laplace(loc=0, scale=sigma/np.sqrt(2), size=n_train)
    y = y_true + laplace_noise
    
    return X, y, y_true, beta, p, sigma

def generate_cauchy_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with Cauchy noise."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    from .snr_utils import _construct_beta_vector
    beta = _construct_beta_vector(p, signal_proportion, 'cauchy', seed=seed)
    y_true = X @ beta
    
    # Generate Cauchy noise
    cauchy_noise = sigma * np.random.standard_cauchy(size=n_train)
    y = y_true + cauchy_noise
    
    return X, y, y_true, beta, p, sigma

