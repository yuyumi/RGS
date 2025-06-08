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


def pve_to_sigma(pve: float, signal_proportion: float, n_predictors: int) -> float:
    """
    Convert PVE (Proportion of Variance Explained) to sigma value.
    
    PVE = (s*p)/(s*p + sigma^2)
    where s is signal_proportion and p is n_predictors
    
    Solving for sigma:
    sigma = sqrt((s*p/PVE) - s*p)
    
    Parameters
    ----------
    pve : float
        Proportion of variance explained
    signal_proportion : float
        Proportion of predictors that are signal
    n_predictors : int
        Total number of predictors
        
    Returns
    -------
    float
        Corresponding sigma value
    """
    sp = signal_proportion * n_predictors
    return np.sqrt((sp/pve) - sp)


def get_sigma_list(sigma_params: Dict[str, Any], 
                   signal_proportion: float, 
                   n_predictors: int) -> List[float]:
    """
    Get list of sigma values based on parameters.
    
    Parameters
    ----------
    sigma_params : dict
        Dictionary containing either:
        - type: "list" and values with list of sigma values
        - type: "pve" and params with num_points, min_pve, max_pve
    signal_proportion : float
        Signal proportion (needed for PVE calculation)
    n_predictors : int
        Number of predictors (needed for PVE calculation)
        
    Returns
    -------
    list
        List of sigma values to use
    """
    if sigma_params['type'] == 'pve':
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
        return [pve_to_sigma(pve, signal_proportion, n_predictors) 
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