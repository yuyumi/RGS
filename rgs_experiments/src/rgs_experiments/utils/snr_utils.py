"""
Signal-to-Noise Ratio (SNR) utilities for simulation experiments.

This module provides functions for computing signal strength, converting between
different SNR representations, and handling various data generator types.
"""

import numpy as np
from typing import Union, Dict, Any, Optional


def pve_to_sigma(pve: float, signal_strength: float) -> float:
    """
    Convert proportion of variance explained (PVE) to noise standard deviation.
    
    PVE = signal_strength / (signal_strength + sigma^2)
    Solving for sigma: sigma = sqrt(signal_strength * (1 - PVE) / PVE)
    
    Parameters
    ----------
    pve : float
        Proportion of variance explained (0 < PVE < 1)
    signal_strength : float
        Signal strength ||X*beta||^2 / n
        
    Returns
    -------
    float
        Noise standard deviation sigma
    """
    if not (0 < pve < 1):
        raise ValueError("PVE must be between 0 and 1")
    
    if signal_strength <= 0:
        raise ValueError("Signal strength must be positive")
    
    return np.sqrt(signal_strength * (1 - pve) / pve)


def sigma_to_pve(sigma: float, signal_strength: float) -> float:
    """
    Convert noise standard deviation to proportion of variance explained (PVE).
    
    PVE = signal_strength / (signal_strength + sigma^2)
    
    Parameters
    ----------
    sigma : float
        Noise standard deviation
    signal_strength : float
        Signal strength ||X*beta||^2 / n
        
    Returns
    -------
    float
        Proportion of variance explained
    """
    if sigma < 0:
        raise ValueError("Sigma must be non-negative")
    
    if signal_strength <= 0:
        raise ValueError("Signal strength must be positive")
    
    return signal_strength / (signal_strength + sigma**2)


def compute_snr(signal_strength: float, sigma: float) -> float:
    """
    Compute signal-to-noise ratio.
    
    SNR = signal_strength / sigma^2
    
    Parameters
    ----------
    signal_strength : float
        Signal strength ||X*beta||^2 / n
    sigma : float
        Noise standard deviation
        
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    return signal_strength / (sigma**2)


def snr_to_sigma(snr: float, signal_strength: float) -> float:
    """
    Convert SNR to noise standard deviation.
    
    SNR = signal_strength / sigma^2
    Solving for sigma: sigma = sqrt(signal_strength / SNR)
    
    Parameters
    ----------
    snr : float
        Signal-to-noise ratio
    signal_strength : float
        Signal strength ||X*beta||^2 / n
        
    Returns
    -------
    float
        Noise standard deviation
    """
    if snr <= 0:
        raise ValueError("SNR must be positive")
    
    if signal_strength <= 0:
        raise ValueError("Signal strength must be positive")
    
    return np.sqrt(signal_strength / snr)


def compute_variance_explained(signal_strength: float, sigma: float) -> float:
    """
    Compute proportion of variance explained.
    
    This is equivalent to sigma_to_pve but with a more descriptive name.
    
    Parameters
    ----------
    signal_strength : float
        Signal strength ||X*beta||^2 / n
    sigma : float
        Noise standard deviation
        
    Returns
    -------
    float
        Proportion of variance explained
    """
    return sigma_to_pve(sigma, signal_strength)


def get_signal_strength_from_results(results_df, method: str = "from_params", params_file_path: str = None) -> Union[float, np.ndarray]:
    """
    Extract or compute signal strength from simulation results DataFrame.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from simulation
    method : str, default="from_params"
        Method to compute signal strength:
        - "from_params": Compute from parameter file (recommended)
        - "from_beta": Use true_beta column if available
        - "from_snr": Use SNR and sigma columns if available
    params_file_path : str, optional
        Path to the parameter file. If None, will try to infer from results file path.
        
    Returns
    -------
    float or np.ndarray
        Signal strength value(s)
        
    Raises
    ------
    ValueError
        If the required data is not available for the specified method
    """
    if method == "from_params":
        if params_file_path is None:
            raise ValueError("params_file_path must be provided when using method='from_params'")
        
        import json
        from pathlib import Path
        
        # Load parameter file
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        
        # Extract parameters needed for signal strength computation
        p = params['data']['n_predictors']
        signal_proportion = params['data']['signal_proportion']
        generator_type = params['data']['generator_type']
        eta = params['data']['generator_params'].get('eta', 0.5)
        seed = params['simulation']['base_seed']
        covariance_type = params['data']['covariance_type']
        
        # Use the unified signal strength computation (now in this module)
        
        # Construct target covariance matrix (theoretical, not from data)
        if covariance_type == 'banded':
            rho = params['data']['banded_params'].get('rho', 0.5)
            target_covariance = _construct_banded_covariance_matrix(p, rho)
        elif covariance_type == 'block':
            block_size = params['data']['block_params'].get('block_size', 20)
            within_correlation = params['data']['block_params'].get('within_correlation', 0.25)
            target_covariance = _construct_block_covariance_matrix(p, block_size, within_correlation)
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}. Supported types: 'banded', 'block'")
        
        # Use the unified computation for all generator types
        return float(compute_expected_signal_strength(
            target_covariance, signal_proportion, generator_type, eta, seed
        ))
    
    elif method == "from_beta":
        raise ValueError("Method 'from_beta' is not supported. Use 'from_params' to compute signal strength with proper covariance structure, or 'from_snr' if SNR data is available.")
    
    elif method == "from_snr":
        if "snr" not in results_df.columns or "sigma" not in results_df.columns:
            raise ValueError("snr and sigma columns not found in results DataFrame. Cannot compute signal strength from SNR.")
        
        # Compute from SNR and sigma
        return results_df["snr"] * (results_df["sigma"]**2)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'from_params', 'from_beta', 'from_snr'")


def _construct_banded_covariance_matrix(p: int, rho: float) -> np.ndarray:
    """
    Construct a banded (AR(1)) covariance matrix with correlation parameter rho.
    
    The matrix has the structure Σ[i,j] = ρ^|i-j|, which is the AR(1) autoregressive 
    correlation structure.
    
    Parameters
    ----------
    p : int
        Size of the covariance matrix (p x p)
    rho : float
        Correlation parameter (0 < rho < 1)
        
    Returns
    -------
    np.ndarray
        Banded covariance matrix of shape (p, p)
    """
    indices = np.arange(p)
    distances = np.abs(indices[:, np.newaxis] - indices)
    return rho**distances


def _construct_block_covariance_matrix(p: int, block_size: int, within_correlation: float) -> np.ndarray:
    """
    Construct a block covariance matrix with specified block size and within-block correlation.
    
    The matrix has blocks of size `block_size` with `within_correlation` between variables 
    within the same block and 0 correlation between variables in different blocks.
    
    Uses modulo assignment to match the data generation logic: variables with the same 
    (i % num_blocks) are in the same block.
    
    Parameters
    ----------
    p : int
        Size of the covariance matrix (p x p)
    block_size : int
        Size of each block
    within_correlation : float
        Correlation between variables within the same block
        
    Returns
    -------
    np.ndarray
        Block covariance matrix of shape (p, p)
    """
    cov_matrix = np.eye(p)
    
    # Calculate number of blocks (must match data generation logic)
    num_blocks = p // block_size
    
    # Fill with block structure - ones on diagonal and within_correlation for same block
    # Use modulo assignment: variables with same (i % num_blocks) are in same block
    for i in range(p):
        for j in range(p):
            if i != j and i % num_blocks == j % num_blocks:  # Same block (modulo grouping)
                cov_matrix[i, j] = within_correlation
    
    return cov_matrix 


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