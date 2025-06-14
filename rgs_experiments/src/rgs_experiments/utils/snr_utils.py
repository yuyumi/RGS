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
        
        # For nonlinear DGP, use the specialized computation from sim_util_dgs
        if generator_type == 'nonlinear':
            # Import the correct function to avoid circular imports
            from .sim_util_dgs import compute_expected_signal_strength
            
            # Construct target covariance matrix
            if covariance_type == 'banded':
                rho = params['data']['banded_params'].get('rho', 0.5)
                target_covariance = _construct_banded_covariance_matrix(p, rho)
            elif covariance_type == 'block':
                block_size = params['data']['block_params'].get('block_size', 20)
                within_correlation = params['data']['block_params'].get('within_correlation', 0.25)
                target_covariance = _construct_block_covariance_matrix(p, block_size, within_correlation)
            else:
                raise ValueError(f"Unsupported covariance_type: {covariance_type}. Supported types: 'banded', 'block'")
            
            # Use the correct nonlinear signal strength computation
            return float(compute_expected_signal_strength(
                target_covariance, signal_proportion, generator_type, eta, seed
            ))
        
        # For all other generator types, use the linear β^T Σ β approach
        else:
            # Construct the true beta vector using the same logic as the simulation
            beta = _construct_beta_vector(p, signal_proportion, generator_type, eta=eta, seed=seed)
            
            # Compute true signal strength β^T Σ β based on covariance structure
            if covariance_type == 'banded':
                rho = params['data']['banded_params'].get('rho', 0.5)
                # Construct banded covariance matrix and compute β^T Σ β
                cov_matrix = _construct_banded_covariance_matrix(p, rho)
                return float(beta.T @ cov_matrix @ beta)
            
            elif covariance_type == 'block':
                block_size = params['data']['block_params'].get('block_size', 20)
                within_correlation = params['data']['block_params'].get('within_correlation', 0.25)
                # Construct block covariance matrix and compute β^T Σ β
                cov_matrix = _construct_block_covariance_matrix(p, block_size, within_correlation)
                return float(beta.T @ cov_matrix @ beta)
            
            else:
                raise ValueError(f"Unsupported covariance_type: {covariance_type}. Supported types: 'banded', 'block'")
    
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
    cov_matrix = np.zeros((p, p))
    
    # Fill the matrix with AR(1) structure: Σ[i,j] = ρ^|i-j|
    for i in range(p):
        for j in range(p):
            cov_matrix[i, j] = rho ** abs(i - j)
    
    return cov_matrix


def _construct_block_covariance_matrix(p: int, block_size: int, within_correlation: float) -> np.ndarray:
    """
    Construct a block covariance matrix with specified block size and within-block correlation.
    
    The matrix has blocks of size `block_size` with `within_correlation` between variables 
    within the same block and 0 correlation between variables in different blocks.
    
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
    
    # Create blocks with within-block correlations
    n_blocks = (p + block_size - 1) // block_size  # Ceiling division
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, p)
        
        # Set within-block correlations
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:  # Off-diagonal elements within block
                    cov_matrix[i, j] = within_correlation
    
    return cov_matrix


def _construct_beta_vector(p, signal_proportion, generator_type='exact', eta=0.5, seed=123):
    """
    Construct coefficient vector beta for different generator types.
    
    This is a copy of the function from sim_util_dgs.py to avoid circular imports.
    
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