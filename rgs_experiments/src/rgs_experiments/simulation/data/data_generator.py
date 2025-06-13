"""
Data generation utilities for simulation experiments.

This module provides a clean interface for generating design matrices
and response data for different experimental scenarios.
"""

import numpy as np
from typing import Dict, Any, Tuple, Callable

# Import from the existing rgs_experiments package
from rgs_experiments.utils.sim_util_dgs import (
    generate_banded_X,
    generate_block_X,
    generate_exact_sparsity_example,
    generate_spaced_sparsity_example,
    generate_inexact_sparsity_example,
    generate_nonlinear_example,
    generate_laplace_example,
    generate_cauchy_example
)


class DataGenerator:
    """
    Manages data generation for simulation experiments.
    
    This class provides a clean interface to the various data generation
    functions and handles the complexity of different parameter combinations.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the data generator with experiment parameters.
        
        Parameters
        ----------
        params : dict
            Complete experiment parameters dictionary
        """
        self.params = params
        self.data_params = params['data']
        
        # Set up the design matrix generator
        self._setup_design_generator()
        
        # Set up the response generator
        self._setup_response_generator()
    
    def _setup_design_generator(self) -> None:
        """Set up the design matrix generator function."""
        covariance_type = self.data_params['covariance_type']
        
        if covariance_type == 'banded':
            banded_params = self.data_params['banded_params']
            self.design_generator = generate_banded_X
            self.design_kwargs = {
                'gamma': banded_params['gamma'],
                'fixed_design': banded_params.get('fixed_design', True)
            }
        
        elif covariance_type == 'block':
            block_params = self.data_params['block_params']
            self.design_generator = generate_block_X
            self.design_kwargs = {
                'block_size': block_params['block_size'],
                'within_correlation': block_params['within_correlation'],
                'fixed_design': block_params.get('fixed_design', True)
            }
        
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")
    
    def _setup_response_generator(self) -> None:
        """Set up the response generator function."""
        generator_type = self.data_params['generator_type']
        
        generators = {
            'exact': generate_exact_sparsity_example,
            'spaced': generate_spaced_sparsity_example,
            'inexact': generate_inexact_sparsity_example,
            'nonlinear': generate_nonlinear_example,
            'laplace': generate_laplace_example,
            'cauchy': generate_cauchy_example
        }
        
        if generator_type not in generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        self.response_generator = generators[generator_type]
        
        # Check if this generator needs eta parameter
        self.needs_eta = generator_type in ['nonlinear', 'inexact']
    
    def generate_design_matrix(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the base design matrix.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        X : ndarray
            Design matrix
        cov_matrix : ndarray
            True covariance matrix (if applicable)
        """
        result = self.design_generator(
            n_predictors=self.data_params['n_predictors'],
            n_train=self.data_params['n_train'],
            seed=seed,
            **self.design_kwargs
        )
        
        # Handle different return formats
        if isinstance(result, tuple):
            return result  # (X, cov_matrix)
        else:
            return result, None  # Just X, no covariance matrix
    
    def generate_response_data(self, X: np.ndarray, sigma: float, 
                             seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
        """
        Generate response data for given design matrix.
        
        Parameters
        ----------
        X : ndarray
            Design matrix
        sigma : float
            Noise level
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        X : ndarray
            Design matrix (potentially modified)
        y : ndarray
            Observed response
        y_true : ndarray
            True response (without noise)
        beta_true : ndarray
            True coefficient vector
        p : int
            Number of predictors
        sigma : float
            Noise level used
        """
        kwargs = {
            'signal_proportion': self.data_params['signal_proportion'],
            'sigma': sigma,
            'seed': seed
        }
        
        # Add eta parameter if needed
        if self.needs_eta:
            generator_params = self.data_params.get('generator_params', {})
            kwargs['eta'] = generator_params.get('eta', 0.5)
        
        return self.response_generator(X, **kwargs)
    
    def generate_train_data(self, X: np.ndarray, sigma: float, 
                           seed: int) -> Dict[str, np.ndarray]:
        """
        Generate training data.
        
        Parameters
        ----------
        X : ndarray
            Base design matrix
        sigma : float
            Noise level
        seed : int
            Random seed
            
        Returns
        -------
        dict
            Dictionary containing training data arrays
        """
        n_train = self.data_params['n_train']
        
        # Create training matrix of appropriate size
        X_train_base = np.zeros((n_train, X.shape[1]))
        X_train_base[:] = X[:n_train] if n_train <= X.shape[0] else X
        
        X_train, y_train, y_true_train, beta_true, p, sigma_used = self.generate_response_data(
            X_train_base, sigma, seed
        )
        
        return {
            'X': X_train,
            'y': y_train,
            'y_true': y_true_train,
            'beta_true': beta_true,
            'p': p,
            'sigma': sigma_used
        }
    
    def generate_test_data(self, X: np.ndarray, sigma: float, 
                          seed: int) -> Dict[str, np.ndarray]:
        """
        Generate test data.
        
        Parameters
        ----------
        X : ndarray
            Base design matrix
        sigma : float
            Noise level
        seed : int
            Random seed (should be different from training seed)
            
        Returns
        -------
        dict
            Dictionary containing test data arrays
        """
        n_test = self.data_params.get('n_test', self.data_params['n_train'])
        
        # Create test matrix of appropriate size
        X_test_base = np.zeros((n_test, X.shape[1]))
        X_test_base[:] = X[:n_test] if n_test <= X.shape[0] else X
        
        X_test, y_test, y_true_test, _, _, _ = self.generate_response_data(
            X_test_base, sigma, seed
        )
        
        return {
            'X': X_test,
            'y': y_test,
            'y_true': y_true_test
        }
    
    def generate_validation_data(self, X: np.ndarray, sigma: float, 
                               seed: int) -> Dict[str, np.ndarray]:
        """
        Generate validation data.
        
        Parameters
        ----------
        X : ndarray
            Base design matrix
        sigma : float
            Noise level
        seed : int
            Random seed (should be different from training/test seeds)
            
        Returns
        -------
        dict
            Dictionary containing validation data arrays
        """
        n_val = self.data_params.get('n_val', self.data_params['n_train'])
        
        # Create validation matrix of appropriate size
        X_val_base = np.zeros((n_val, X.shape[1]))
        X_val_base[:] = X[:n_val] if n_val <= X.shape[0] else X
        
        X_val, y_val, y_true_val, _, _, _ = self.generate_response_data(
            X_val_base, sigma, seed
        )
        
        return {
            'X': X_val,
            'y': y_val,
            'y_true': y_true_val
        } 