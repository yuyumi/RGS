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
    'generate_spaced_sparsity_example',
    '_compute_nonlinear_signal_strength_empirical',
    'compute_expected_signal_strength'
]

# Helper functions to reduce redundancy
def _construct_beta_vector(p, signal_proportion, generator_type='exact', eta=0.5, seed=123):
    """
    Construct coefficient vector beta for different generator types.
    
    Parameters:
    -----------
    p : int
        Number of predictors
    signal_proportion : float
        Proportion of variables that are signals
    generator_type : str
        Type of generator ('exact', 'inexact', 'nonlinear', 'laplace', 'cauchy', 'spaced')
    eta : float
        Parameter for inexact/nonlinear generators
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    beta : ndarray
        Coefficient vector
    """
    np.random.seed(seed)
    signals = int(p * signal_proportion)
    
    if generator_type in ['exact', 'laplace', 'cauchy']:
        # Linear: β = [1, 1, ..., 1, 0, 0, ..., 0]
        beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
        
    elif generator_type == 'spaced':
        # Evenly spaced signals
        beta = np.zeros(p)
        if signals > 0:
            if signals == 1:
                beta[p // 2] = 1
            elif signals >= p:
                beta = np.ones(p)
            else:
                indices = np.linspace(0, p - 1, signals)
                indices = np.round(indices).astype(int)
                beta[indices] = 1
                
    elif generator_type == 'inexact':
        # Exponentially decaying coefficients with alternating signs
        beta = np.zeros(p)
        beta[:signals] = 1.0  # Strong signals
        
        # Weak signals: β_i = (-1)^i * exp(-i*eta) for s < i ≤ p (using 1-based indexing)
        for i in range(signals+1, p+1):  # 1-based indexing: s+1 to p
            idx = i - 1  # Convert to 0-based indexing
            beta[idx] = ((-1)**i) * np.exp(-i*eta)
        
    elif generator_type == 'nonlinear':
        # For nonlinear, return the linear component beta
        beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
        
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")
    
    return beta

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



def generate_banded_X(n_predictors, n_train, rho=0.65, seed=123, fixed_design=True):
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
    fixed_design : bool, default=True
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

def generate_block_X(n_predictors, n_train, block_size, within_correlation=0.7, seed=123, fixed_design=True):
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
    fixed_design : bool, default=True
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
    
    beta = _construct_beta_vector(p, signal_proportion, 'exact', seed=seed)
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_spaced_sparsity_example(X, signal_proportion, sigma=None, seed=123):
    """Generate example with evenly spaced signals across all predictors."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
    beta = _construct_beta_vector(p, signal_proportion, 'spaced', seed=seed)
    y_true = X @ beta
    y = y_true + np.random.normal(0, sigma, n_train)

    return X, y, y_true, beta, p, sigma

def generate_inexact_sparsity_example(X, signal_proportion, sigma=None, eta=0.5, seed=123):
    """Generate example with inexact sparsity."""
    _validate_sigma(sigma)
    np.random.seed(seed)
    n_train, p = X.shape
    
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
    
    beta = _construct_beta_vector(p, signal_proportion, 'cauchy', seed=seed)
    y_true = X @ beta
    
    # Generate Cauchy noise
    cauchy_noise = sigma * np.random.standard_cauchy(size=n_train)
    y = y_true + cauchy_noise
    
    return X, y, y_true, beta, p, sigma

def _compute_nonlinear_signal_strength_empirical(X, signal_proportion, eta=0.5, seed=123):
    """
    Compute empirical signal strength for nonlinear generator type only.
    
    This function is only used internally for Monte Carlo estimation in 
    compute_expected_signal_strength() for the nonlinear case.
    
    For nonlinear signals, use the formula:
        signal_i = eta * (sum_{j=1}^{s2} x_{ij}^2 + sum_{j=1}^{s2-1} sum_{l=j+1}^{s2} x_{ij} x_{il}) / sqrt(p)
                  + (1-eta) * sum_{j=1}^{s} x_{ij}
    where s = number of signals, s2 = s//2, p = number of predictors
    Signal strength is then mean(signal_i^2) over all samples.
    
    Note: This function now includes the √p scaling factor to match generate_nonlinear_example.
    
    Parameters:
    -----------
    X : ndarray
        Design matrix
    signal_proportion : float
        Proportion of variables that are signals
    eta : float
        Parameter for nonlinear generator
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    signal_strength : float
        Average squared magnitude of the nonlinear signal (||signal||^2/n)
    """
    np.random.seed(seed)
    n_train, p = X.shape
    signals = int(p * signal_proportion)
    
    # Only handle nonlinear case
    s = signals
    s2 = s // 2
    signal = np.zeros(n_train)
    for i in range(n_train):
        x_row = X[i, :]
        # Nonlinear part: sum of squares
        quad_sum = np.sum(x_row[:s2] ** 2)
        # Nonlinear part: pairwise products - only if we have at least 2 nonlinear variables
        pair_sum = 0.0
        if s2 >= 2:  # Need at least 2 variables for interactions
            for j in range(s2 - 1):
                for l in range(j + 1, s2):
                    pair_sum += x_row[j] * x_row[l]
        # Apply √p scaling to nonlinear component (to match generate_nonlinear_example)
        nonlinear_component = (quad_sum + pair_sum) / np.sqrt(p)
        # Linear part
        linear_sum = np.sum(x_row[:s])
        # Combine
        signal[i] = eta * nonlinear_component + (1 - eta) * linear_sum
    # Compute signal strength as mean squared signal
    signal_strength = np.mean(signal ** 2)
    
    return signal_strength

def compute_expected_signal_strength(target_covariance, signal_proportion, generator_type='exact', eta=0.5, seed=123):
    """
    Compute the expected signal strength for random designs using the population covariance matrix.
    
    For random designs, we use the target covariance matrix directly rather than a realized design matrix.
    Signal strength is defined as E[||Xβ||²/n] = βᵀΣβ, which is the expected average squared magnitude
    of the signal. This is used to compute SNR = signal_strength/σ².
    
    Parameters:
    -----------
    target_covariance : ndarray
        Target population covariance matrix
    signal_proportion : float
        Proportion of variables that are signals
    generator_type : str
        Type of generator ('exact', 'inexact', 'nonlinear', 'laplace', 'cauchy', 'spaced')
    eta : float
        Parameter for inexact/nonlinear generators
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    expected_signal_strength : float
        Expected average squared magnitude of the signal (βᵀΣβ)
    """
    np.random.seed(seed)
    p = target_covariance.shape[0]
    signals = int(p * signal_proportion)
    
    if generator_type == 'nonlinear':
        # For nonlinear case with random design, use Monte Carlo estimation
        # The analytical computation is extremely complex due to higher-order moments
        # of multivariate normal distributions. Monte Carlo gives reliable results.
        
        # Generate a small number of realizations to estimate expected signal strength
        np.random.seed(seed)
        n_monte_carlo = 50  # Reasonable number for stable estimates
        signal_strengths = []
        
        for i in range(n_monte_carlo):
            # Generate random design matrix with target covariance
            from scipy.stats import multivariate_normal
            # Use a reasonable sample size for signal strength estimation
            n_samples = max(1000, 10 * p)  
            X_sample = multivariate_normal.rvs(
                mean=np.zeros(p),
                cov=target_covariance,
                size=n_samples,
                random_state=seed + i
            )
            
            # Compute signal strength for this realization
            strength = _compute_nonlinear_signal_strength_empirical(
                X_sample, signal_proportion, eta, seed
            )
            signal_strengths.append(strength)
        
        # Return the average signal strength
        expected_signal_strength = np.mean(signal_strengths)
        return expected_signal_strength
        
    else:
        # For all other generator types, use the helper function
        beta = _construct_beta_vector(p, signal_proportion, generator_type, eta=eta, seed=seed)
    
    # For linear cases, expected signal strength is βᵀΣβ
    expected_signal_strength = beta.T @ target_covariance @ beta
    
    return expected_signal_strength