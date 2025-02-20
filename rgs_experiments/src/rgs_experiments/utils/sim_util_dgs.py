import numpy as np
from scipy.stats import ortho_group

__all__ = [
    'generate_orthogonal_X',
    'generate_banded_X',
    'generate_block_X',
    'generate_exact_sparsity_example',
    'generate_inexact_sparsity_example',
    'generate_nonlinear_example',
    'generate_laplace_example',
    'generate_cauchy_example'
]

def generate_orthogonal_X(n_predictors, n_train, seed=123):
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

def generate_banded_X(n_predictors, n_train, gamma=0.65, seed=123):
    """
    Generate a design matrix X with AR(1)-like correlation structure.
    Each block is transformed by a different random orthogonal matrix.
    """
    np.random.seed(seed)
    
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
    X = X * np.sqrt(n_predictors)
    
    # Verify gram matrix structure
    realized_gram = X.T @ X/(n_train)
    max_diff = np.max(np.abs(realized_gram - gram_matrix))
    print(f"Maximum deviation from target in gram matrix: {max_diff:.2e}")
    
    return X

def generate_block_X(n_predictors, n_train, block_size, within_correlation=0.7, seed=123):
    """
    Generate a design matrix X with block correlation structure.
    """
    np.random.seed(seed)
    
    # Verify parameters
    if n_predictors % block_size != 0:
        raise ValueError("n_predictors must be divisible by block_size")
    
    n_blocks = n_predictors // block_size
    
    # Create block correlation matrix
    gram_matrix = np.zeros((n_predictors, n_predictors))
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        block = np.ones((block_size, block_size)) * within_correlation
        np.fill_diagonal(block, 1.0)  # Set diagonal to 1
        gram_matrix[start_idx:end_idx, start_idx:end_idx] = block
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
    
    if np.any(eigenvals < -1e-10):
        raise ValueError("Correlation matrix has negative eigenvalues")
    
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
    
    # Scale to match target covariance
    X = X * np.sqrt(n_predictors)
    
    # Verify gram matrix structure
    realized_gram = X.T @ X/(n_train)
    max_diff = np.max(np.abs(realized_gram - gram_matrix))
    print(f"Maximum deviation from target in gram matrix: {max_diff:.2e}")
    
    return X

def generate_exact_sparsity_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with exact sparsity."""
    assert sigma is not None, "sigma parameter must be provided"
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_inexact_sparsity_example(X, signal_proportion, sigma=None, eta=0.5, seed=123):
    """Generate example with inexact sparsity."""
    assert sigma is not None, "sigma parameter must be provided"
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

def generate_nonlinear_example(X, signal_proportion, sigma=None, eta=0.5, seed=123):
    """Generate example with nonlinear components."""
    assert sigma is not None, "sigma parameter must be provided"
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    # Linear component
    beta_linear = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    linear_signal = X @ beta_linear
    
    # Nonlinear component using interactions and quadratic terms
    X_active = X[:, :signals//2]
    quad_terms = X_active**2
    
    # Create pairwise interactions
    interactions = np.zeros(n_train)
    for i in range(4):
        for j in range(i+1, 5):
            interactions += X_active[:, i] * X_active[:, j]
    
    # Combine linear and nonlinear components
    nonlinear_signal = (quad_terms.sum(axis=1) + interactions) / np.sqrt(p)
    y_true = (1 - eta) * linear_signal + eta * nonlinear_signal
    y = y_true + np.random.normal(0, sigma, n_train)
    
    return X, y, y_true, beta_linear, p, sigma

def generate_laplace_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with Laplace noise."""
    assert sigma is not None, "sigma parameter must be provided"
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    y_true = X @ beta
    
    # Generate Laplace noise (scale = sigma/sqrt(2) to match N(0,sigma^2) variance)
    laplace_noise = np.random.laplace(loc=0, scale=sigma/np.sqrt(2), size=n_train)
    y = y_true + laplace_noise
    
    return X, y, y_true, beta, p, sigma

def generate_cauchy_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with Cauchy noise."""
    assert sigma is not None, "sigma parameter must be provided"
    np.random.seed(seed)
    n_train,p = X.shape
    signals = int(p*signal_proportion)
    
    beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
    y_true = X @ beta
    
    # Generate Cauchy noise
    cauchy_noise = sigma * np.random.standard_cauchy(size=n_train)
    y = y_true + cauchy_noise
    
    return X, y, y_true, beta, p, sigma