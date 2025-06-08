"""
Matrix utilities for simulation experiments.

This module provides utilities for checking matrix properties
and diagnosing potential numerical issues.
"""

import numpy as np
from typing import Dict, Union


def check_matrix_rank(X: np.ndarray) -> Dict[str, Union[bool, int, float]]:
    """
    Check if a matrix X is full rank and provide diagnostic information.
    
    Parameters
    ----------
    X : ndarray
        The design matrix to check
        
    Returns
    -------
    dict
        Dictionary containing rank information and diagnostics
    """
    
    n, p = X.shape
    min_dim = min(n, p)
    
    # Calculate rank using SVD
    # SVD is more numerically stable than np.linalg.matrix_rank
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    
    # Get the rank (number of non-zero singular values, with tolerance)
    tol = s.max() * max(X.shape) * np.finfo(s.dtype).eps
    rank = np.sum(s > tol)
    
    # Calculate condition number
    condition_number = s[0] / s[-1] if s[-1] > tol else np.inf
    
    # Check if X'X is positive definite
    is_pos_def = True
    try:
        # Try Cholesky decomposition - will fail if not positive definite
        np.linalg.cholesky(X.T @ X)
    except np.linalg.LinAlgError:
        is_pos_def = False
    
    # Prepare results
    results = {
        'is_full_rank': bool(rank == min_dim),  # Convert from numpy.bool_ to Python bool
        'rank': int(rank),  # Convert from numpy.int64 to Python int
        'min_dimension': int(min_dim),
        'condition_number': float(condition_number),
        'smallest_singular_value': float(s[-1]),
        'largest_singular_value': float(s[0]),
        'XTX_is_positive_definite': bool(is_pos_def)
    }
    return results


def print_matrix_diagnostics(X: np.ndarray, matrix_name: str = "X") -> None:
    """
    Print comprehensive matrix diagnostics.
    
    Parameters
    ----------
    X : ndarray
        Matrix to analyze
    matrix_name : str
        Name of the matrix for display purposes
    """
    rank_info = check_matrix_rank(X)
    
    print(f"Rank check for matrix {matrix_name}:")
    print(f"  - Dimensions: {X.shape}")
    print(f"  - Full rank: {rank_info['is_full_rank']}")
    print(f"  - Rank: {rank_info['rank']} / {rank_info['min_dimension']}")
    print(f"  - Condition number: {rank_info['condition_number']:.4e}")
    
    # If not full rank, provide more detailed information
    if not rank_info['is_full_rank']:
        print(f"WARNING: {matrix_name} is not full rank!")
        print(f"  - Smallest singular value: {rank_info['smallest_singular_value']:.4e}")
        print(f"  - X'X is positive definite: {rank_info['XTX_is_positive_definite']}")
        print("  - This may lead to unstable or non-unique solutions.")


def check_near_constant_features(X: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """
    Check for features with near-zero variance that could cause numerical issues.
    
    Parameters
    ----------
    X : ndarray
        Design matrix
    threshold : float
        Threshold below which features are considered near-constant
        
    Returns
    -------
    ndarray
        Indices of near-constant features
    """
    X_centered = X - X.mean(axis=0)
    feature_norms = np.sqrt(np.sum(X_centered**2, axis=0))
    near_zero_features = np.where(feature_norms < threshold)[0]
    
    if len(near_zero_features) > 0:
        print(f"WARNING: Found {len(near_zero_features)} near-constant features:")
        print(f"  Feature indices: {list(near_zero_features)}")
        print(f"  Consider removing these features before fitting.")
    
    return near_zero_features 