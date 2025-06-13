"""
Signal-to-Noise Ratio (SNR) utilities for simulation experiments.

This module provides functions for computing signal strength, converting between
different SNR representations, and handling various data generator types.
"""

import numpy as np
from typing import Union, Dict, Any, Optional


def compute_signal_strength(beta: np.ndarray, X: Optional[np.ndarray] = None, 
                          generator_type: str = "normal", 
                          generator_params: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute the signal strength ||X*beta||^2 / n for different generator types.
    
    For most generators, this equals ||beta||^2. For correlated generators,
    it accounts for the covariance structure.
    
    Parameters
    ----------
    beta : np.ndarray
        True coefficient vector
    X : np.ndarray, optional
        Design matrix. Required for certain generator types.
    generator_type : str, default="normal"
        Type of data generator used
    generator_params : dict, optional
        Parameters for the data generator
        
    Returns
    -------
    float
        Signal strength
    """
    if generator_type in ["normal", "uniform", "exponential", "laplace"]:
        # For independent generators, signal strength is ||beta||^2
        return np.sum(beta**2)
    
    elif generator_type == "correlated_normal":
        if X is None:
            # Fallback to ||beta||^2 if X not provided
            return np.sum(beta**2)
        
        # For correlated data, compute actual signal strength
        signal = X @ beta
        return np.mean(signal**2)
    
    elif generator_type == "banded":
        # For banded correlation structure
        if generator_params is None:
            rho = 0.5  # default
        else:
            rho = generator_params.get("rho", 0.5)
        
        # Approximate signal strength for banded correlation
        # This is an approximation - exact computation would require the full covariance matrix
        base_strength = np.sum(beta**2)
        
        # Account for correlation between adjacent features
        correlation_adjustment = 0.0
        for i in range(len(beta) - 1):
            correlation_adjustment += 2 * rho * beta[i] * beta[i+1]
        
        return base_strength + correlation_adjustment
    
    else:
        # Default fallback
        return np.sum(beta**2)


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
        
        # Extract parameters needed for beta construction
        p = params['data']['n_predictors']
        signal_proportion = params['data']['signal_proportion']
        generator_type = params['data']['generator_type']
        eta = params['data']['generator_params'].get('eta', 0.5)
        seed = params['simulation']['base_seed']
        
        # Construct the true beta vector using the same logic as the simulation
        beta = _construct_beta_vector(p, signal_proportion, generator_type, eta=eta, seed=seed)
        
        # For banded covariance, we need to account for correlation structure
        if params['data']['covariance_type'] == 'banded':
            gamma = params['data']['banded_params'].get('gamma', 0.5)
            # Approximate signal strength accounting for correlation
            base_strength = np.sum(beta**2)
            correlation_adjustment = 0.0
            for i in range(len(beta) - 1):
                correlation_adjustment += 2 * gamma * beta[i] * beta[i+1]
            return base_strength + correlation_adjustment
        else:
            # For other covariance types, use simple ||beta||^2
            return np.sum(beta**2)
    
    elif method == "from_beta":
        if "true_beta" not in results_df.columns:
            raise ValueError("true_beta column not found in results DataFrame. Cannot compute signal strength from beta values.")
        
        # Compute from true beta values
        signal_strengths = []
        for beta_str in results_df["true_beta"]:
            if isinstance(beta_str, str):
                # Parse string representation of array
                beta = np.fromstring(beta_str.strip("[]"), sep=" ")
            else:
                beta = np.array(beta_str)
            signal_strengths.append(np.sum(beta**2))
        return np.array(signal_strengths)
    
    elif method == "from_snr":
        if "snr" not in results_df.columns or "sigma" not in results_df.columns:
            raise ValueError("snr and sigma columns not found in results DataFrame. Cannot compute signal strength from SNR.")
        
        # Compute from SNR and sigma
        return results_df["snr"] * (results_df["sigma"]**2)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'from_params', 'from_beta', 'from_snr'")


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
        # Exponentially decaying coefficients
        beta = np.zeros(p)
        beta[:signals] = 1.0  # Strong signals
        
        # Generate random signs for remaining coefficients
        signs = np.random.choice([-1, 1], size=p-signals)
        indices = np.arange(p-signals)
        magnitudes = np.exp(-eta * indices)
        beta[signals:] = signs * magnitudes
        
    elif generator_type == 'nonlinear':
        # For nonlinear, return the linear component beta
        beta = np.concatenate([np.full(signals, 1), np.zeros(p-signals)])
        
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")
    
    return beta 