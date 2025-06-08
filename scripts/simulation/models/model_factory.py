"""
Model factory for creating different types of feature selection models.

This module provides a factory pattern for creating and configuring
different types of feature selection models used in the simulations.
"""

import time
from typing import Dict, Any, Callable, Tuple
import numpy as np

# Import from core RGS package
from rgs.core.rgs import RGS, RGSCV
from rgs.mse import create_mse_scorer

# Import from simulation package
from rgs_experiments.models.data_smearing import SmearedGS
from rgs_experiments.models.bagging import BaggedGS

# Import parameter utilities
from ..config.parameter_loader import get_m_grid


class ModelFactory:
    """
    Factory for creating and fitting different types of feature selection models.
    
    This class encapsulates the model creation logic that was originally
    scattered throughout simulation_main.py.
    """
    
    def __init__(self, params: Dict[str, Any], cv_value: int, make_k_scorer: Callable):
        """
        Initialize the model factory.
        
        Parameters
        ----------
        params : dict
            Simulation parameters containing model configuration
        cv_value : int
            Cross-validation fold count
        make_k_scorer : callable
            Function to create k-specific scorers
        """
        self.params = params
        self.cv_value = cv_value
        self.make_k_scorer = make_k_scorer
        
    def create_bagged_gs(self, seed: int, sim_num: int) -> Tuple[BaggedGS, float]:
        """
        Create and fit a BaggedGS model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = BaggedGS(
            k_max=self.params['model']['k_max'],
            n_estimators=self.params['model']['bagged_gs']['n_estimators'],
            random_state=seed + sim_num,
            method=self.params['model'].get('method', 'fs'),
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_smeared_gs(self, seed: int, sim_num: int) -> Tuple[SmearedGS, float]:
        """
        Create and fit a SmearedGS model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = SmearedGS(
            k_max=self.params['model']['k_max'],
            n_estimators=self.params['model']['smeared_gs']['n_estimators'],
            noise_scale=self.params['model']['smeared_gs']['param_grid']['noise_scale'],
            random_state=seed + sim_num,
            method=self.params['model'].get('method', 'fs'),
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_rgscv(self, seed: int, sim_num: int) -> Tuple[RGSCV, float]:
        """
        Create and fit an RGSCV model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        # Calculate m_grid
        m_grid = get_m_grid(
            self.params['model']['m_grid'],
            self.params['data']['n_predictors']
        )
        
        model = RGSCV(
            k_max=self.params['model']['k_max'],
            m_grid=m_grid,
            n_estimators=self.params['model']['rgscv']['n_estimators'],
            n_resample_iter=self.params['model']['rgscv']['n_resample_iter'],
            method=self.params['model'].get('method', 'fs'),
            random_state=seed + sim_num,
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_greedy_selection(self, seed: int, sim_num: int) -> Tuple[RGSCV, float]:
        """
        Create and fit a Greedy Selection model (RGSCV with specific parameters).
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = RGSCV(
            k_max=self.params['model']['k_max'],
            m_grid=[self.params['data']['n_predictors']],  # Use all features
            n_estimators=1,
            n_resample_iter=0,
            method=self.params['model'].get('method', 'fs'),
            random_state=seed + sim_num,
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def fit_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                  start_time: float) -> Tuple[Any, float]:
        """
        Fit a model and return it with timing information.
        
        Parameters
        ----------
        model : Any
            Model to fit
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets
        start_time : float
            Time when model creation started
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        model.fit(X_train, y_train)
        fitting_time = time.time() - start_time
        return model, fitting_time 