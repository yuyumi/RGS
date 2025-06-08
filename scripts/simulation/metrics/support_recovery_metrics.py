"""
Support recovery evaluation metrics.

This module provides functions for evaluating how well
models recover the true support set of non-zero coefficients.
"""

import numpy as np


def calculate_f_score(beta_hat: np.ndarray, 
                     beta_true: np.ndarray, 
                     threshold: float = 1e-10) -> float:
    """
    Calculate F-score for support recovery.
    
    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients
    beta_true : ndarray
        True coefficients
    threshold : float
        Threshold for considering a coefficient non-zero
        
    Returns
    -------
    float
        F-score value (harmonic mean of precision and recall)
    """
    # Identify non-zero coefficients (true support)
    true_support = (np.abs(beta_true) > threshold)
    
    # Identify coefficients identified as non-zero by the model
    predicted_support = (np.abs(beta_hat) > threshold)
    
    # Calculate true positives, false positives, false negatives
    true_positives = np.sum(predicted_support & true_support)
    false_positives = np.sum(predicted_support & ~true_support)
    false_negatives = np.sum(~predicted_support & true_support)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_score 