"""
Degrees of freedom calculations for model evaluation.

This module provides functions for calculating degrees of freedom
for different types of models using the Stein formula.
"""

import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import mean_squared_error


def calculate_df_for_all_k(model: Any, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray, 
                          y_true_train: np.ndarray, 
                          sigma: float, 
                          n_train: int) -> Dict[int, float]:
    """
    Calculate degrees of freedom for each k value in a fitted RGS model.
    
    Parameters
    ----------
    model : RGS or RGSCV
        The fitted model with coefficients for each k
    X_train : ndarray
        Training features
    y_train : ndarray
        Observed training targets
    y_true_train : ndarray
        True training targets (without noise)
    sigma : float
        Noise standard deviation
    n_train : int
        Number of training samples
        
    Returns
    -------
    dict
        Dictionary mapping k values to their degrees of freedom
    """
    # For RGSCV models, extract the underlying RGS model
    if hasattr(model, 'model_'):
        rgs_model = model.model_
    else:
        rgs_model = model
    
    # Calculate df for each k value
    df_by_k = {}
    
    for k in range(len(rgs_model.coef_)):
        # Get predictions for this k
        y_pred_k = rgs_model.predict(X_train, k=k)
        
        # Calculate MSE (observed vs. predicted)
        mse_k = np.mean((y_train - y_pred_k) ** 2)
        
        # Calculate in-sample error (true signal vs. predicted)
        insample_k = np.mean((y_true_train - y_pred_k) ** 2)
        
        # Calculate degrees of freedom using the formula
        error_diff_k = insample_k - mse_k + sigma**2
        df_k = (n_train / (2 * sigma**2)) * error_diff_k
        
        # Store in dictionary
        df_by_k[k] = df_k
    
    return df_by_k


def calculate_df_for_all_k_ensemble(model: Any, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray, 
                                   y_true_train: np.ndarray, 
                                   sigma: float, 
                                   n_train: int) -> Dict[int, float]:
    """
    Calculate degrees of freedom for each k value in a fitted ensemble model.
    
    Parameters
    ----------
    model : BaggedGS or SmearedGS
        The fitted ensemble model
    X_train : ndarray
        Training features
    y_train : ndarray
        Observed training targets
    y_true_train : ndarray
        True training targets (without noise)
    sigma : float
        Noise standard deviation
    n_train : int
        Number of training samples
        
    Returns
    -------
    dict
        Dictionary mapping k values to their degrees of freedom
    """
    df_by_k = {}
    
    # Get maximum k value
    k_max = model.k_max
    
    for k in range(k_max + 1):
        # Get average coefficients for this k
        avg_coef_k = np.zeros(X_train.shape[1])
        count = 0
        
        for coefs, _, _ in model.estimators_:
            if k < len(coefs):  # Ensure k is valid for this estimator
                avg_coef_k += coefs[k]
                count += 1
        
        if count > 0:  # Only proceed if we have valid estimators
            avg_coef_k /= count
            
            # Calculate the average intercept
            avg_intercept = np.mean(y_train) - X_train.mean(axis=0) @ avg_coef_k
            
            # Make predictions
            y_pred_k = X_train @ avg_coef_k + avg_intercept
            
            # Calculate errors
            mse_k = np.mean((y_train - y_pred_k) ** 2)
            insample_k = np.mean((y_true_train - y_pred_k) ** 2)
            
            # Calculate df
            error_diff_k = insample_k - mse_k + sigma**2
            df_k = (n_train / (2 * sigma**2)) * error_diff_k
            
            df_by_k[k] = df_k
    
    return df_by_k


def calculate_mse_for_all_k(model: Any, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate MSE for each k value in a fitted model.
    
    Parameters
    ----------
    model : Model with predict method
        The fitted model
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
        
    Returns
    -------
    dict
        Dictionary mapping k values to their MSE
    """
    # For RGSCV models, extract the underlying RGS model
    if hasattr(model, 'model_'):
        rgs_model = model.model_
    else:
        rgs_model = model
    
    mse_by_k = {}
    
    for k in range(len(rgs_model.coef_)):
        # Get predictions for this k
        y_pred_k = rgs_model.predict(X_train, k=k)
        
        # Calculate MSE
        mse_k = mean_squared_error(y_train, y_pred_k)
        
        # Store in dictionary
        mse_by_k[k] = mse_k
    
    return mse_by_k


def calculate_insample_for_all_k(model: Any, 
                                X_train: np.ndarray, 
                                y_true_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate in-sample error for each k value in a fitted model.
    
    Parameters
    ----------
    model : Model with predict method
        The fitted model
    X_train : ndarray
        Training features
    y_true_train : ndarray
        True training targets (without noise)
        
    Returns
    -------
    dict
        Dictionary mapping k values to their in-sample error
    """
    # For RGSCV models, extract the underlying RGS model
    if hasattr(model, 'model_'):
        rgs_model = model.model_
    else:
        rgs_model = model
    
    insample_by_k = {}
    
    for k in range(len(rgs_model.coef_)):
        # Get predictions for this k
        y_pred_k = rgs_model.predict(X_train, k=k)
        
        # Calculate in-sample error
        insample_k = mean_squared_error(y_true_train, y_pred_k)
        
        # Store in dictionary
        insample_by_k[k] = insample_k
    
    return insample_by_k


def calculate_mse_for_all_k_ensemble(model: Any, 
                                    X_train: np.ndarray, 
                                    y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate MSE for each k value in a fitted ensemble model.
    
    Parameters
    ----------
    model : Ensemble model with predict method
        The fitted ensemble model
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
        
    Returns
    -------
    dict
        Dictionary mapping k values to their MSE
    """
    mse_by_k = {}
    
    # Get maximum k value
    k_max = model.k_max
    
    for k in range(k_max + 1):
        # Get average coefficients for this k
        avg_coef_k = np.zeros(X_train.shape[1])
        count = 0
        
        for coefs, _, _ in model.estimators_:
            if k < len(coefs):  # Ensure k is valid for this estimator
                avg_coef_k += coefs[k]
                count += 1
        
        if count > 0:  # Only proceed if we have valid estimators
            avg_coef_k /= count
            
            # Calculate the average intercept
            avg_intercept = np.mean(y_train) - X_train.mean(axis=0) @ avg_coef_k
            
            # Make predictions
            y_pred_k = X_train @ avg_coef_k + avg_intercept
            
            # Calculate MSE
            mse_k = np.mean((y_train - y_pred_k) ** 2)
            
            mse_by_k[k] = mse_k
    
    return mse_by_k


def calculate_insample_for_all_k_ensemble(model: Any, 
                                         X_train: np.ndarray, 
                                         y_true_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate in-sample error for each k value in a fitted ensemble model.
    
    Parameters
    ----------
    model : Ensemble model with predict method
        The fitted ensemble model
    X_train : ndarray
        Training features
    y_true_train : ndarray
        True training targets (without noise)
        
    Returns
    -------
    dict
        Dictionary mapping k values to their in-sample error
    """
    insample_by_k = {}
    
    # Get maximum k value
    k_max = model.k_max
    
    for k in range(k_max + 1):
        # Get average coefficients for this k
        avg_coef_k = np.zeros(X_train.shape[1])
        count = 0
        
        for coefs, _, _ in model.estimators_:
            if k < len(coefs):  # Ensure k is valid for this estimator
                avg_coef_k += coefs[k]
                count += 1
        
        if count > 0:  # Only proceed if we have valid estimators
            avg_coef_k /= count
            
            # Calculate the average intercept
            avg_intercept = np.mean(y_true_train) - X_train.mean(axis=0) @ avg_coef_k
            
            # Make predictions
            y_pred_k = X_train @ avg_coef_k + avg_intercept
            
            # Calculate in-sample error
            insample_k = np.mean((y_true_train - y_pred_k) ** 2)
            
            insample_by_k[k] = insample_k
    
    return insample_by_k


def calculate_single_model_df(mse: float, 
                             insample_error: float, 
                             sigma: float, 
                             n_train: int) -> float:
    """
    Calculate degrees of freedom for a single model using Stein's formula.
    
    Parameters
    ----------
    mse : float
        Mean squared error (observed vs predicted)
    insample_error : float
        In-sample error (true signal vs predicted)
    sigma : float
        Noise standard deviation
    n_train : int
        Number of training samples
        
    Returns
    -------
    float
        Degrees of freedom estimate
    """
    error_diff = insample_error - mse + sigma**2
    df = (n_train / (2 * sigma**2)) * error_diff
    return df 