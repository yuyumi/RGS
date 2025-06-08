"""
Experiment orchestrator for coordinating single experiment runs.

This module provides the ExperimentOrchestrator class that coordinates
all the refactored components to run a complete experiment iteration.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer

# Import our refactored components
from ..data import DataGenerator
from ..models import ModelFactory, ModelEvaluator
from ..config.parameter_loader import get_sigma_list

# Import scoring utilities
from rgs.mse import create_mse_scorer


class ExperimentOrchestrator:
    """
    Orchestrates a complete single experiment iteration.
    
    This class coordinates all the refactored components to run a complete
    experiment that was previously handled by run_one_dgp_iter.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the experiment orchestrator.
        
        Parameters
        ----------
        params : dict
            Complete experiment parameters
        """
        self.params = params
        self.data_generator = DataGenerator(params)
        
        # Check validation approach
        self.use_validation_set = params.get('simulation', {}).get('use_validation_set', False)
        
    def _setup_cross_validation(self, X_val: Optional[np.ndarray] = None, 
                               y_val: Optional[np.ndarray] = None,
                               sigma: float = None) -> Tuple[int, Any]:
        """
        Set up cross-validation strategy and scoring function.
        
        Parameters
        ----------
        X_val : ndarray, optional
            Validation features (if using validation set approach)
        y_val : ndarray, optional
            Validation targets (if using validation set approach)
        sigma : float
            Noise level
            
        Returns
        -------
        tuple
            (cv_value, make_k_scorer_function)
        """
        if self.use_validation_set and X_val is not None and y_val is not None:
            # Validation set approach
            cv_value = 1
            
            def create_validation_scorer(X_val, y_val):
                def make_k_scorer(k):
                    def validation_score(y_true, y_pred):
                        # Note: y_true is ignored, we always use y_val
                        return -mean_squared_error(y_val, y_pred)
                    return make_scorer(validation_score)
                return make_k_scorer
            
            make_k_scorer = create_validation_scorer(X_val, y_val)
        else:
            # Standard CV approach
            cv_value = self.params['model']['baseline']['cv']
            make_k_scorer = create_mse_scorer(
                sigma=sigma,
                n=self.params['data']['n_train'],
                p=self.params['data']['n_predictors']
            )
        
        return cv_value, make_k_scorer
    
    def _fit_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           y_true_train: np.ndarray, beta_true: np.ndarray,
                           cov_matrix: np.ndarray, sigma: float, seed: int, sim_num: int,
                           X_val: Optional[np.ndarray] = None, 
                           y_val: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Fit and evaluate baseline models (Lasso, Ridge, ElasticNet).
        
        Parameters
        ----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets
        X_test : ndarray
            Test features
        y_test : ndarray
            Test targets
        y_true_train : ndarray
            True training targets (without noise)
        beta_true : ndarray
            True coefficients
        cov_matrix : ndarray
            Covariance matrix
        sigma : float
            Noise level
        seed : int
            Random seed
        sim_num : int
            Simulation number
        X_val : ndarray, optional
            Validation features
        y_val : ndarray, optional
            Validation targets
            
        Returns
        -------
        tuple
            (results_dict, timing_dict)
        """
        from ..metrics import calculate_relative_test_error, calculate_f_score
        
        results = {}
        timing_results = {}
        n_train = len(y_train)
        
        def select_model_by_validation(models, params_list):
            """Select best model based on validation performance."""
            val_scores = []
            for model in models:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_score = mean_squared_error(y_val, val_pred)
                val_scores.append(val_score)
            best_idx = np.argmin(val_scores)
            best_model = models[best_idx]
            best_params = params_list[best_idx]
            return best_model, best_params
        
        # Get CV value for baseline models
        cv_value = 1 if self.use_validation_set else self.params['model']['baseline']['cv']
        
        # Fit Lasso
        start_time = time.time()
        if self.use_validation_set and X_val is not None:
            alphas = np.logspace(-10, 1, 100)
            lasso_models = [Lasso(alpha=alpha, random_state=seed+sim_num) for alpha in alphas]
            lasso, best_alpha = select_model_by_validation(lasso_models, alphas)
        else:
            lasso = LassoCV(cv=cv_value, random_state=seed+sim_num)
            lasso.fit(X_train, y_train)
        timing_results['time_lasso'] = time.time() - start_time
        
        # Evaluate Lasso
        y_pred_lasso = lasso.predict(X_train)
        y_test_lasso = lasso.predict(X_test)
        
        mse_lasso = mean_squared_error(y_train, y_pred_lasso)
        insample_lasso = mean_squared_error(y_true_train, y_pred_lasso)
        error_diff_lasso = insample_lasso - mse_lasso + sigma**2
        df_lasso = (n_train / (2 * sigma**2)) * error_diff_lasso
        
        rte_lasso = calculate_relative_test_error(
            beta_hat=lasso.coef_, beta_true=beta_true,
            X_test=X_test, sigma=sigma, cov_matrix=cov_matrix
        )
        f_score_lasso = calculate_f_score(beta_hat=lasso.coef_, beta_true=beta_true)
        
        results.update({
            'insample_lasso': insample_lasso,
            'mse_lasso': mse_lasso,
            'df_lasso': df_lasso,
            'coef_recovery_lasso': np.mean((lasso.coef_ - beta_true)**2),
            'support_recovery_lasso': np.mean((lasso.coef_ != 0) == (beta_true != 0)),
            'outsample_mse_lasso': mean_squared_error(y_test, y_test_lasso),
            'rte_lasso': rte_lasso,
            'f_score_lasso': f_score_lasso
        })
        
        # Fit Ridge
        start_time = time.time()
        if self.use_validation_set and X_val is not None:
            alphas = np.logspace(-10, 10, 100)
            ridge_models = [Ridge(alpha=alpha) for alpha in alphas]
            ridge, best_alpha = select_model_by_validation(ridge_models, alphas)
        else:
            ridge = RidgeCV(cv=cv_value)
            ridge.fit(X_train, y_train)
        timing_results['time_ridge'] = time.time() - start_time
        
        # Evaluate Ridge
        y_pred_ridge = ridge.predict(X_train)
        y_test_ridge = ridge.predict(X_test)
        
        mse_ridge = mean_squared_error(y_train, y_pred_ridge)
        insample_ridge = mean_squared_error(y_true_train, y_pred_ridge)
        error_diff_ridge = insample_ridge - mse_ridge + sigma**2
        df_ridge = (n_train / (2 * sigma**2)) * error_diff_ridge
        
        rte_ridge = calculate_relative_test_error(
            beta_hat=ridge.coef_, beta_true=beta_true,
            X_test=X_test, sigma=sigma, cov_matrix=cov_matrix
        )
        f_score_ridge = calculate_f_score(beta_hat=ridge.coef_, beta_true=beta_true)
        
        results.update({
            'insample_ridge': insample_ridge,
            'mse_ridge': mse_ridge,
            'df_ridge': df_ridge,
            'coef_recovery_ridge': np.mean((ridge.coef_ - beta_true)**2),
            'outsample_mse_ridge': mean_squared_error(y_test, y_test_ridge),
            'rte_ridge': rte_ridge,
            'f_score_ridge': f_score_ridge
        })
        
        # Fit ElasticNet
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        start_time = time.time()
        if self.use_validation_set and X_val is not None:
            alphas = np.logspace(-10, 1, 20)
            elastic_params = [(alpha, l1) for alpha in alphas for l1 in l1_ratios]
            elastic_models = [ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed+sim_num) 
                             for alpha, l1 in elastic_params]
            elastic, best_elastic_params = select_model_by_validation(elastic_models, elastic_params)
        else:
            elastic = ElasticNetCV(l1_ratio=l1_ratios, cv=cv_value, random_state=seed+sim_num)
            elastic.fit(X_train, y_train)
        timing_results['time_elastic'] = time.time() - start_time
        
        # Evaluate ElasticNet
        y_pred_elastic = elastic.predict(X_train)
        y_test_elastic = elastic.predict(X_test)
        
        mse_elastic = mean_squared_error(y_train, y_pred_elastic)
        insample_elastic = mean_squared_error(y_true_train, y_pred_elastic)
        error_diff_elastic = insample_elastic - mse_elastic + sigma**2
        df_elastic = (n_train / (2 * sigma**2)) * error_diff_elastic
        
        rte_elastic = calculate_relative_test_error(
            beta_hat=elastic.coef_, beta_true=beta_true,
            X_test=X_test, sigma=sigma, cov_matrix=cov_matrix
        )
        f_score_elastic = calculate_f_score(beta_hat=elastic.coef_, beta_true=beta_true)
        
        results.update({
            'insample_elastic': insample_elastic,
            'mse_elastic': mse_elastic,
            'df_elastic': df_elastic,
            'coef_recovery_elastic': np.mean((elastic.coef_ - beta_true)**2),
            'support_recovery_elastic': np.mean((elastic.coef_ != 0) == (beta_true != 0)),
            'outsample_mse_elastic': mean_squared_error(y_test, y_test_elastic),
            'rte_elastic': rte_elastic,
            'f_score_elastic': f_score_elastic
        })
        
        return results, timing_results
    
    def _fit_rfs_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       y_true_train: np.ndarray, beta_true: np.ndarray,
                       cov_matrix: np.ndarray, sigma: float, seed: int, sim_num: int,
                       cv_value: int, make_k_scorer: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Fit and evaluate RFS models (BaggedGS, SmearedGS, RGSCV, Greedy Selection).
        
        Parameters
        ----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets
        X_test : ndarray
            Test features
        y_test : ndarray
            Test targets
        y_true_train : ndarray
            True training targets (without noise)
        beta_true : ndarray
            True coefficients
        cov_matrix : ndarray
            Covariance matrix
        sigma : float
            Noise level
        seed : int
            Random seed
        sim_num : int
            Simulation number
        cv_value : int
            Cross-validation value
        make_k_scorer : callable
            Scoring function factory
            
        Returns
        -------
        tuple
            (results_dict, timing_dict)
        """
        results = {}
        timing_results = {}
        
        # Initialize model factory and evaluator
        model_factory = ModelFactory(self.params, cv_value, make_k_scorer)
        model_evaluator = ModelEvaluator(sigma=sigma, n_train=len(y_train))
        
        # Fit BaggedGS
        bagged_model, start_time = model_factory.create_bagged_gs(seed, sim_num)
        bagged_model, fitting_time = model_factory.fit_model(bagged_model, X_train, y_train, start_time)
        timing_results['time_bagged_gs'] = fitting_time
        
        bagged_results = model_evaluator.evaluate_model_complete(
            model=bagged_model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            X_test=X_test, y_test=y_test, beta_true=beta_true, cov_matrix=cov_matrix,
            model_name="bagged_gs", is_ensemble=True
        )
        results.update(bagged_results)
        
        # Fit SmearedGS
        smeared_model, start_time = model_factory.create_smeared_gs(seed, sim_num)
        smeared_model, fitting_time = model_factory.fit_model(smeared_model, X_train, y_train, start_time)
        timing_results['time_smeared_gs'] = fitting_time
        
        smeared_results = model_evaluator.evaluate_model_complete(
            model=smeared_model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            X_test=X_test, y_test=y_test, beta_true=beta_true, cov_matrix=cov_matrix,
            model_name="smeared_gs", is_ensemble=True
        )
        results.update(smeared_results)
        
        # Fit RGSCV
        rgscv_model, start_time = model_factory.create_rgscv(seed, sim_num)
        rgscv_model, fitting_time = model_factory.fit_model(rgscv_model, X_train, y_train, start_time)
        timing_results['time_rgscv'] = fitting_time
        
        rgscv_results = model_evaluator.evaluate_model_complete(
            model=rgscv_model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            X_test=X_test, y_test=y_test, beta_true=beta_true, cov_matrix=cov_matrix,
            model_name="rgs", is_ensemble=False
        )
        results.update(rgscv_results)
        
        # Fit Greedy Selection
        gs_model, start_time = model_factory.create_greedy_selection(seed, sim_num)
        gs_model, fitting_time = model_factory.fit_model(gs_model, X_train, y_train, start_time)
        timing_results['time_original_gs'] = fitting_time
        
        gs_results = model_evaluator.evaluate_model_complete(
            model=gs_model, X_train=X_train, y_train=y_train, y_true_train=y_true_train,
            X_test=X_test, y_test=y_test, beta_true=beta_true, cov_matrix=cov_matrix,
            model_name="gs", is_ensemble=False
        )
        results.update(gs_results)
        
        return results, timing_results
    
    def run_single_experiment(self, X: np.ndarray, cov_matrix: np.ndarray,
                            sigma: float, seed: int, sim_num: int) -> Dict[str, Any]:
        """
        Run a single complete experiment iteration.
        
        This is the refactored version of run_one_dgp_iter that coordinates
        all the modularized components.
        
        Parameters
        ----------
        X : ndarray
            Base design matrix
        cov_matrix : ndarray
            Covariance matrix
        sigma : float
            Noise level
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        dict
            Complete results dictionary
        """
        # Initialize result
        results = {
            'simulation': sim_num,
            'sigma': sigma,
            'method': self.params['model'].get('method', 'fs')
        }
        timing_results = {'simulation': sim_num, 'sigma': sigma}
        
        # Generate training data
        train_data = self.data_generator.generate_train_data(X, sigma, seed + sim_num)
        X_train = train_data['X']
        y_train = train_data['y']
        y_true_train = train_data['y_true']
        beta_true = train_data['beta_true']
        
        # Generate test data
        test_data = self.data_generator.generate_test_data(X, sigma, seed + sim_num + 200000)
        X_test = test_data['X']
        y_test = test_data['y']
        
        # Generate validation data if needed
        X_val = None
        y_val = None
        if self.use_validation_set:
            val_data = self.data_generator.generate_validation_data(X, sigma, seed + sim_num + 100000)
            X_val = val_data['X']
            y_val = val_data['y']
        
        # Set up cross-validation strategy
        cv_value, make_k_scorer = self._setup_cross_validation(X_val, y_val, sigma)
        
        # Fit baseline models
        baseline_results, baseline_timing = self._fit_baseline_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            y_true_train=y_true_train, beta_true=beta_true, cov_matrix=cov_matrix,
            sigma=sigma, seed=seed, sim_num=sim_num, X_val=X_val, y_val=y_val
        )
        results.update(baseline_results)
        timing_results.update(baseline_timing)
        
        # Fit RFS models
        rfs_results, rfs_timing = self._fit_rfs_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            y_true_train=y_true_train, beta_true=beta_true, cov_matrix=cov_matrix,
            sigma=sigma, seed=seed, sim_num=sim_num, cv_value=cv_value, make_k_scorer=make_k_scorer
        )
        results.update(rfs_results)
        timing_results.update(rfs_timing)
        
        # Add timing results to main results
        results.update(timing_results)
        
        return results 