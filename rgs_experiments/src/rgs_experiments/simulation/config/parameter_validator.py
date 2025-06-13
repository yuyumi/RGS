"""
Parameter validation utilities.

This module validates simulation parameters to catch configuration
errors early and provide helpful error messages.
"""

from typing import Dict, Any, List
import warnings


def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate simulation parameters and return validated/processed params.
    
    Parameters
    ----------
    params : dict
        Raw parameters loaded from JSON
        
    Returns
    -------
    dict
        Validated and potentially modified parameters
        
    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent
    """
    validated_params = params.copy()
    
    # Validate simulation parameters
    _validate_simulation_params(validated_params['simulation'])
    
    # Validate data parameters
    _validate_data_params(validated_params['data'])
    
    # Validate model parameters
    _validate_model_params(validated_params['model'])
    
    # Validate output parameters
    _validate_output_params(validated_params['output'])
    
    # Cross-validate parameters
    _validate_parameter_combinations(validated_params)
    
    return validated_params


def _validate_simulation_params(sim_params: Dict[str, Any]) -> None:
    """Validate simulation-level parameters."""
    required_keys = ['n_sim', 'base_seed', 'sigma']
    for key in required_keys:
        if key not in sim_params:
            raise ValueError(f"Missing required simulation parameter: {key}")
    
    if sim_params['n_sim'] <= 0:
        raise ValueError("n_sim must be positive")
    
    if not isinstance(sim_params['base_seed'], int):
        raise ValueError("base_seed must be an integer")


def _validate_data_params(data_params: Dict[str, Any]) -> None:
    """Validate data generation parameters."""
    required_keys = ['n_predictors', 'n_train', 'signal_proportion', 
                     'covariance_type', 'generator_type']
    for key in required_keys:
        if key not in data_params:
            raise ValueError(f"Missing required data parameter: {key}")
    
    # Validate dimensions
    if data_params['n_predictors'] <= 0:
        raise ValueError("n_predictors must be positive")
    
    if data_params['n_train'] <= 0:
        raise ValueError("n_train must be positive")
    
    # Validate signal proportion
    if not 0 < data_params['signal_proportion'] <= 1:
        raise ValueError("signal_proportion must be between 0 and 1")
    
    # Validate covariance type
    valid_cov_types = ['banded', 'block']
    if data_params['covariance_type'] not in valid_cov_types:
        raise ValueError(f"covariance_type must be one of {valid_cov_types}")
    
    # Validate generator type
    valid_gen_types = ['exact', 'spaced', 'inexact', 'nonlinear', 'laplace', 'cauchy']
    if data_params['generator_type'] not in valid_gen_types:
        raise ValueError(f"generator_type must be one of {valid_gen_types}")
    
    # Validate generator-specific parameters
    generators_needing_eta = ['inexact', 'nonlinear']
    if data_params['generator_type'] in generators_needing_eta:
        if 'generator_params' not in data_params:
            raise ValueError(f"generator_params required for generator_type '{data_params['generator_type']}'")
        if 'eta' not in data_params['generator_params']:
            raise ValueError(f"eta parameter required in generator_params for generator_type '{data_params['generator_type']}'")
        eta = data_params['generator_params']['eta']
        if not 0 <= eta <= 1:
            raise ValueError("eta must be between 0 and 1")
    
    # Validate type-specific parameters
    if data_params['covariance_type'] == 'banded':
        if 'banded_params' not in data_params:
            raise ValueError("banded_params required for banded covariance")
        banded_params = data_params['banded_params']
        if 'gamma' not in banded_params:
            raise ValueError("gamma required in banded_params")
        gamma = banded_params['gamma']
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        
        # Validate fixed_design parameter if present
        if 'fixed_design' in banded_params:
            if not isinstance(banded_params['fixed_design'], bool):
                raise ValueError("fixed_design must be a boolean")
    
    if data_params['covariance_type'] == 'block':
        if 'block_params' not in data_params:
            raise ValueError("block_params required for block covariance")
        block_params = data_params['block_params']
        required_block_keys = ['block_size', 'within_correlation']
        for key in required_block_keys:
            if key not in block_params:
                raise ValueError(f"{key} required in block_params")
        
        # Validate fixed_design parameter if present
        if 'fixed_design' in block_params:
            if not isinstance(block_params['fixed_design'], bool):
                raise ValueError("fixed_design must be a boolean")
        
        if data_params['n_predictors'] % block_params['block_size'] != 0:
            raise ValueError("n_predictors must be divisible by block_size")


def _validate_model_params(model_params: Dict[str, Any]) -> None:
    """Validate model parameters."""
    required_keys = ['k_max', 'm_grid']
    for key in required_keys:
        if key not in model_params:
            raise ValueError(f"Missing required model parameter: {key}")
    
    if model_params['k_max'] <= 0:
        raise ValueError("k_max must be positive")
    
    # Validate method if present
    if 'method' in model_params:
        valid_methods = ['fs', 'omp']
        if model_params['method'] not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")


def _validate_output_params(output_params: Dict[str, Any]) -> None:
    """Validate output parameters."""
    if 'save_path' not in output_params:
        raise ValueError("Missing required output parameter: save_path")


def _validate_parameter_combinations(params: Dict[str, Any]) -> None:
    """Validate cross-parameter constraints."""
    data_params = params['data']
    model_params = params['model']
    
    # k_max shouldn't exceed n_predictors
    if model_params['k_max'] > data_params['n_predictors']:
        warnings.warn(
            f"k_max ({model_params['k_max']}) is greater than n_predictors "
            f"({data_params['n_predictors']}). This may cause issues."
        )
    
    # n_train should be reasonable relative to n_predictors
    if data_params['n_train'] < data_params['n_predictors']:
        warnings.warn(
            f"n_train ({data_params['n_train']}) is less than n_predictors "
            f"({data_params['n_predictors']}). This is a high-dimensional setting."
        ) 