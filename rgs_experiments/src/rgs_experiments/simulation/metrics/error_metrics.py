"""
Error and loss metrics for model evaluation.

This module provides functions for calculating various error
and loss metrics used to evaluate model performance.
"""

import numpy as np
from typing import Union


def calculate_relative_test_error(beta_hat: np.ndarray, 
                                beta_true: np.ndarray, 
                                X_test: np.ndarray, 
                                sigma: float, 
                                cov_matrix: Union[np.ndarray, None] = None) -> float:
    """
    Calculate the Relative Test Error (RTE).
    
    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients
    beta_true : ndarray
        True coefficients
    X_test : ndarray
        Test design matrix
    sigma : float
        Noise standard deviation
    cov_matrix : ndarray, optional
        The covariance matrix of X. If None, it will be estimated from X_test.
        
    Returns
    -------
    float
        The relative test error (RTE)
    """
    # Calculate the covariance matrix if not provided
    if cov_matrix is None:
        cov_matrix = np.cov(X_test, rowvar=False)
    
    # Calculate the covariance-weighted squared difference between true and estimated betas
    beta_diff = beta_hat - beta_true
    weighted_beta_diff = beta_diff.T @ cov_matrix @ beta_diff
    
    # The RTE formula: (beta_hat - beta_true)^T Σ (beta_hat - beta_true) + sigma^2) / sigma^2
    rte = (weighted_beta_diff + sigma**2) / (sigma**2)
    
    return rte


def calculate_relative_insample_error(beta_hat: np.ndarray, 
                                    beta_true: np.ndarray, 
                                    X_train: np.ndarray, 
                                    sigma: float, 
                                    cov_matrix: Union[np.ndarray, None] = None) -> float:
    """
    Calculate the Relative In-sample Error (RIE).
    
    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients
    beta_true : ndarray
        True coefficients
    X_train : ndarray
        Training design matrix
    sigma : float
        Noise standard deviation
    cov_matrix : ndarray, optional
        The covariance matrix of X. If None, it will be estimated from X_train.
        
    Returns
    -------
    float
        The relative in-sample error (RIE)
    """
    # Calculate the covariance matrix if not provided
    if cov_matrix is None:
        cov_matrix = np.cov(X_train, rowvar=False)
    
    # Calculate the covariance-weighted squared difference between true and estimated betas
    beta_diff = beta_hat - beta_true
    weighted_beta_diff = beta_diff.T @ cov_matrix @ beta_diff
    
    # The RIE formula: (beta_hat - beta_true)^T Σ̂ (beta_hat - beta_true) + sigma^2) / sigma^2
    rie = (weighted_beta_diff + sigma**2) / (sigma**2)
    
    return rie 