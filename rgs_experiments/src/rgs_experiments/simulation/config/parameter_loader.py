"""
Parameter loading and processing utilities.

This module handles loading simulation parameters from JSON files
and converting them into usable formats for experiments.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union


def load_params(param_path: Union[str, Path]) -> Dict[str, Any]:
    """Load parameters from JSON file."""
    with open(param_path, 'r') as f:
        params = json.load(f)
    return params


def pve_to_sigma(pve: float, signal_strength: float) -> float:
    """
    Convert PVE (Proportion of Variance Explained) to sigma value.
    
    PVE = signal_strength / (signal_strength + sigma^2)
    where signal_strength is ||Xβ||²/n (average squared magnitude of signal)
    
    Solving for sigma:
    sigma = sqrt(signal_strength * (1/PVE - 1))
    
    Parameters
    ----------
    pve : float
        Proportion of variance explained
    signal_strength : float
        Average squared magnitude of signal (||Xβ||²/n)
        
    Returns
    -------
    float
        Corresponding sigma value
    """
    if pve <= 0 or pve >= 1:
        raise ValueError(f"PVE must be between 0 and 1, got {pve}")
    
    return np.sqrt(signal_strength * (1/pve - 1))


def get_sigma_list(sigma_params: Dict[str, Any], 
                   target_covariance: np.ndarray,
                   signal_proportion: float,
                   generator_type: str = 'exact',
                   eta: float = 0.5,
                   seed: int = 123) -> List[float]:
    """
    Get list of sigma values based on parameters.
    
    Uses theoretical signal strength β^T Σ β for all designs. This is appropriate because:
    - Fixed design: X is constructed such that (1/n)X^T X ≈ Σ by design
    - Random design: E[X^T X / n] = Σ by definition
    
    Parameters
    ----------
    sigma_params : dict
        Dictionary containing either:
        - type: "list" and values with list of sigma values
        - type: "pve" and params with num_points, min_pve, max_pve
    target_covariance : ndarray
        Target covariance matrix from sim_util_dgs (e.g., from generate_banded_X)
    signal_proportion : float
        Signal proportion (needed for PVE calculation)
    generator_type : str
        Type of generator ('exact', 'inexact', 'nonlinear', etc.)
    eta : float
        Parameter for inexact/nonlinear generators
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of sigma values to use
    """
    if sigma_params['type'] == 'pve':
        # Import here to avoid circular imports
        from rgs_experiments.utils.snr_utils import compute_expected_signal_strength
        
        # Use theoretical signal strength β^T Σ β
        # This works for both fixed and random designs since sim_util_dgs provides the covariance matrix
        signal_strength = compute_expected_signal_strength(
            target_covariance, signal_proportion, generator_type, eta, seed
        )
        
        if sigma_params['style'] == 'list':
            pve_values = sigma_params['values']
        elif sigma_params['style'] == 'range':
            pve_values = np.linspace(
                sigma_params['params']['min'],
                sigma_params['params']['max'],
                sigma_params['params']['num_points']
            )
        else:
            raise ValueError(f"Unknown PVE style: {sigma_params['style']}")
        return [pve_to_sigma(pve, signal_strength) 
                for pve in pve_values]
    elif sigma_params['type'] == 'sigma':
        if sigma_params['style'] == 'list':
            return sigma_params['values']
        elif sigma_params['style'] == 'range':
            return np.linspace(
                sigma_params['params']['min'],
                sigma_params['params']['max'],
                sigma_params['params']['num_points']
            )
        else:
            raise ValueError(f"Unknown sigma style: {sigma_params['style']}")
    else:
        raise ValueError(f"Unknown sigma type: {sigma_params['type']}")


def get_m_grid(grid_params: Dict[str, Any], n_predictors: int) -> List[int]:
    """
    Get m_grid based on parameters.
    
    Parameters
    ----------
    grid_params : dict
        Dictionary containing either:
        - type: "geometric" and params with base and num_points
        - type: "list" and values with list of m values
    n_predictors : int
        Number of predictors (needed for geometric grid)
    
    Returns
    -------
    list
        List of m values to use
    """
    if grid_params['type'] == 'geometric':
        base = grid_params['params']['base']
        num_points = grid_params['params']['num_points']
        return [int(2 + (n_predictors-2) * (base**x - 1)/(base**(num_points-1) - 1)) 
                for x in range(num_points)]
    elif grid_params['type'] == 'list':
        # Simply return the values as a Python list, don't convert to np.array
        return [int(x) for x in grid_params['values']]
    else:
        raise ValueError(f"Unknown m_grid type: {grid_params['type']}") 