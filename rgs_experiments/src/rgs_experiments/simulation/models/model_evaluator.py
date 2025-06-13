"""
Model evaluation utilities for feature selection models.

This module provides functionality for evaluating different types of
feature selection models and computing various performance metrics.
"""

from typing import Dict, Any, Tuple, Union
import numpy as np
from sklearn.metrics import mean_squared_error

# Import metrics
from ..metrics import (
    calculate_relative_test_error,
    calculate_relative_insample_error,
    calculate_f_score,
    calculate_df_for_all_k,
    calculate_df_for_all_k_ensemble,
    calculate_mse_for_all_k,
    calculate_insample_for_all_k,
    calculate_mse_for_all_k_ensemble,
    calculate_insample_for_all_k_ensemble
)


class ModelEvaluator:
    """
    Evaluator for different types of feature selection models.
    
    This class encapsulates the model evaluation logic that was originally
    scattered throughout simulation_main.py.
    """
    
    def __init__(self, sigma: float, n_train: int):
        """
        Initialize the model evaluator.
        
        Parameters
        ----------
        sigma : float
            Noise standard deviation
        n_train : int
            Number of training samples
        """
        self.sigma = sigma
        self.n_train = n_train
    
    def extract_coefficients_regular(self, model: Any) -> np.ndarray:
        """
        Extract coefficients from regular models (RGSCV).
        
        Parameters
        ----------
        model : RGSCV
            Fitted model
            
        Returns
        -------
        ndarray
            Model coefficients at optimal k
        """
        return model.model_.coef_[model.k_]
    
    def extract_coefficients_ensemble(self, model: Any, n_features: int) -> np.ndarray:
        """
        Extract average coefficients from ensemble models (BaggedGS, SmearedGS).
        
        Parameters
        ----------
        model : BaggedGS or SmearedGS
            Fitted ensemble model
        n_features : int
            Number of features
            
        Returns
        -------
        ndarray
            Average coefficients across ensemble
        """
        avg_coef = np.zeros(n_features)
        for coefs, _, _ in model.estimators_:
            avg_coef += coefs[model.k_]
        avg_coef /= len(model.estimators_)
        return avg_coef
    
    def evaluate_model_basic(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            y_true_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                            beta_true: np.ndarray, cov_matrix: np.ndarray,
                            model_name: str, is_ensemble: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model and compute basic metrics.
        
        Parameters
        ----------
        model : Any
            Fitted model to evaluate
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets (with noise)
        y_true_train : ndarray
            Training targets (without noise)
        X_test : ndarray
            Test features
        y_test : ndarray
            Test targets
        beta_true : ndarray
            True coefficients
        cov_matrix : ndarray
            Covariance matrix
        model_name : str
            Name prefix for result keys
        is_ensemble : bool
            Whether the model is an ensemble model
            
        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Extract coefficients
        if is_ensemble:
            coefficients = self.extract_coefficients_ensemble(model, X_train.shape[1])
        else:
            coefficients = self.extract_coefficients_regular(model)
        
        # Calculate basic metrics
        mse = mean_squared_error(y_train, y_pred_train)
        insample_error = mean_squared_error(y_true_train, y_pred_train)
        error_diff = insample_error - mse + self.sigma**2
        df = (self.n_train / (2 * self.sigma**2)) * error_diff
        
        # Calculate RTE, RIE, and F-score
        rte = calculate_relative_test_error(
            beta_hat=coefficients,
            beta_true=beta_true,
            X_test=X_test,
            sigma=self.sigma,
            cov_matrix=cov_matrix
        )
        
        # Calculate sample covariance matrix for RIE
        train_cov_matrix = np.cov(X_train, rowvar=False)
        
        rie = calculate_relative_insample_error(
            beta_hat=coefficients,
            beta_true=beta_true,
            X_train=X_train,
            sigma=self.sigma,
            cov_matrix=train_cov_matrix
        )
        
        f_score = calculate_f_score(
            beta_hat=coefficients,
            beta_true=beta_true
        )
        
        # Compile results
        results = {
            f'best_k_{model_name}': model.k_,
            f'insample_{model_name}': insample_error,
            f'mse_{model_name}': mse,
            f'df_{model_name}': df,
            f'coef_recovery_{model_name}': np.mean((coefficients - beta_true)**2),
            f'support_recovery_{model_name}': np.mean(
                (np.abs(coefficients) > 1e-10) == (beta_true != 0)
            ),
            f'outsample_mse_{model_name}': mean_squared_error(y_test, y_pred_test),
            f'rte_{model_name}': rte,
            f'rie_{model_name}': rie,
            f'f_score_{model_name}': f_score
        }
        
        # Add model-specific parameters
        if hasattr(model, 'm_'):
            results[f'best_m'] = model.m_
        if hasattr(model, 'noise_scale_'):
            results[f'best_noise_scale'] = model.noise_scale_
            
        return results
    
    def evaluate_model_detailed(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                               y_true_train: np.ndarray, model_name: str,
                               is_ensemble: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model with detailed k-wise metrics.
        
        Parameters
        ----------
        model : Any
            Fitted model to evaluate
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets (with noise)
        y_true_train : ndarray
            Training targets (without noise)
        model_name : str
            Name prefix for result keys
        is_ensemble : bool
            Whether the model is an ensemble model
            
        Returns
        -------
        dict
            Dictionary containing detailed k-wise metrics
        """
        results = {}
        
        # Calculate metrics for all k values
        if is_ensemble:
            df_by_k = calculate_df_for_all_k_ensemble(
                model=model, X_train=X_train, y_train=y_train,
                y_true_train=y_true_train, sigma=self.sigma, n_train=self.n_train
            )
            mse_by_k = calculate_mse_for_all_k_ensemble(
                model=model, X_train=X_train, y_train=y_train
            )
            insample_by_k = calculate_insample_for_all_k_ensemble(
                model=model, X_train=X_train, y_true_train=y_true_train
            )
        else:
            df_by_k = calculate_df_for_all_k(
                model=model, X_train=X_train, y_train=y_train,
                y_true_train=y_true_train, sigma=self.sigma, n_train=self.n_train
            )
            mse_by_k = calculate_mse_for_all_k(
                model=model, X_train=X_train, y_train=y_train
            )
            insample_by_k = calculate_insample_for_all_k(
                model=model, X_train=X_train, y_true_train=y_true_train
            )
        
        # Store k-wise results
        for k, mse_value in mse_by_k.items():
            results[f'mse_by_k_{model_name}_{k}'] = mse_value
            
        for k, insample_value in insample_by_k.items():
            results[f'insample_by_k_{model_name}_{k}'] = insample_value
            
        for k, df_value in df_by_k.items():
            results[f'df_by_k_{model_name}_{k}'] = df_value
            
        return results
    
    def evaluate_model_complete(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                               y_true_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                               beta_true: np.ndarray, cov_matrix: np.ndarray,
                               model_name: str, is_ensemble: bool = False) -> Dict[str, Any]:
        """
        Perform complete evaluation of a model (both basic and detailed).
        
        Parameters
        ----------
        model : Any
            Fitted model to evaluate
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets (with noise)
        y_true_train : ndarray
            Training targets (without noise)
        X_test : ndarray
            Test features
        y_test : ndarray
            Test targets
        beta_true : ndarray
            True coefficients
        cov_matrix : ndarray
            Covariance matrix
        model_name : str
            Name prefix for result keys
        is_ensemble : bool
            Whether the model is an ensemble model
            
        Returns
        -------
        dict
            Dictionary containing all evaluation metrics
        """
        # Get basic evaluation
        basic_results = self.evaluate_model_basic(
            model=model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            X_test=X_test, y_test=y_test, beta_true=beta_true, cov_matrix=cov_matrix,
            model_name=model_name, is_ensemble=is_ensemble
        )
        
        # Get detailed evaluation
        detailed_results = self.evaluate_model_detailed(
            model=model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            model_name=model_name, is_ensemble=is_ensemble
        )
        
        # Combine results
        basic_results.update(detailed_results)
        return basic_results 