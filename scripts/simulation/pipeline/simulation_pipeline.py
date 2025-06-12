"""
Simulation pipeline for coordinating complete simulation runs.

This module provides the SimulationPipeline class that coordinates
the complete simulation workflow across multiple parameter combinations.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our refactored components
from ..config.parameter_loader import load_params, get_sigma_list
from ..data import DataGenerator
from ..data.matrix_utils import check_matrix_rank
from ..orchestration import ExperimentOrchestrator


class SimulationPipeline:
    """
    Coordinates complete simulation runs across multiple parameter combinations.
    
    This class provides the top-level interface for running complete simulations.
    """
    
    def __init__(self, param_path: str):
        """
        Initialize the simulation pipeline.
        
        Parameters
        ----------
        param_path : str
            Path to the simulation parameters JSON file
        """
        self.param_path = Path(param_path)
        self.params = load_params(str(param_path))
        self.start_time = None
        
        # Initialize components
        self.data_generator = DataGenerator(self.params)
        self.experiment_orchestrator = ExperimentOrchestrator(self.params)
        
    def _setup_design_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and validate the base design matrix.
        
        Returns
        -------
        tuple
            (design_matrix, covariance_matrix)
        """
        # Generate base design matrix
        X, cov_matrix = self.data_generator.generate_design_matrix(
            seed=self.params['simulation']['base_seed']
        )
        
        # Check matrix rank and log information
        rank_info = check_matrix_rank(X)
        
        print(f"Design Matrix Information:")
        print(f"  - Dimensions: {X.shape}")
        print(f"  - Full rank: {rank_info['is_full_rank']}")
        print(f"  - Rank: {rank_info['rank']} / {rank_info['min_dimension']}")
        print(f"  - Condition number: {rank_info['condition_number']:.4e}")
        
        if not rank_info['is_full_rank']:
            print("WARNING: X is not full rank!")
            print(f"  - Smallest singular value: {rank_info['smallest_singular_value']:.4e}")
            print(f"  - X'X is positive definite: {rank_info['XTX_is_positive_definite']}")
            print("  - This may lead to unstable or non-unique solutions.")
        
        # Add rank information to parameters for saving
        self.params['rank_check'] = rank_info
        
        return X, cov_matrix
    
    def _get_sigma_values(self, X: np.ndarray, cov_matrix: np.ndarray) -> List[float]:
        """
        Generate the list of sigma values for the simulation.
        
        Parameters
        ----------
        X : ndarray
            Design matrix (for fixed designs)
        cov_matrix : ndarray
            Covariance matrix (for random designs)
        
        Returns
        -------
        list
            Sorted list of sigma values
        """
        # Determine if using fixed design
        if self.params['data']['covariance_type'] == 'block':
            fixed_design = self.params['data']['block_params'].get('fixed_design', True)
        else:
            fixed_design = True  # Other covariance types use fixed design
        
        sigmas = get_sigma_list(
            self.params['simulation']['sigma'],
            X=X if fixed_design else None,
            target_covariance=cov_matrix if not fixed_design else None,
            signal_proportion=self.params['data']['signal_proportion'],
            generator_type=self.params['data']['generator_type'],
            eta=self.params['data']['generator_params'].get('eta', 0.5),
            seed=self.params['simulation']['base_seed'],
            fixed_design=fixed_design
        )
        sigmas = sorted(sigmas)  # Sort for nice progression in progress bar
        
        # Save the actual sigma values used
        self.params['simulation']['sigma']['computed_values'] = sigmas
        
        return sigmas
    
    def _run_experiment_grid(self, X: np.ndarray, cov_matrix: np.ndarray,
                           sigmas: List[float]) -> List[Dict[str, Any]]:
        """
        Run the complete experiment grid across simulations and sigma values.
        
        Parameters
        ----------
        X : ndarray
            Base design matrix
        cov_matrix : ndarray
            Covariance matrix
        sigmas : list
            List of sigma values to test
            
        Returns
        -------
        list
            List of all experiment results
        """
        all_results = []
        
        # Main simulation loop with progress bar
        total_sims = self.params['simulation']['n_sim'] * len(sigmas)
        
        with tqdm(total=total_sims, desc="Total Progress") as pbar:
            for sim in range(1, self.params['simulation']['n_sim'] + 1):
                seed = self.params['simulation']['base_seed']
                
                for sigma in sigmas:
                    # Update progress bar description
                    pbar.set_description(
                        f"Sim {sim}/{self.params['simulation']['n_sim']}, "
                        f"σ={sigma:.3f}, {self.params['data']['generator_type']}, "
                        f"{self.params['data']['covariance_type']}"
                    )
                    
                    # Run single experiment
                    result = self.experiment_orchestrator.run_single_experiment(
                        X=X,
                        cov_matrix=cov_matrix,
                        sigma=sigma,
                        seed=seed,
                        sim_num=sim
                    )
                    
                    all_results.append(result)
                    pbar.update(1)
        
        return all_results
    
    def _create_summary_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics from the results.
        
        Parameters
        ----------
        results_df : DataFrame
            Raw results from all experiments
            
        Returns
        -------
        DataFrame
            Summary statistics grouped by sigma
        """
        # Define metrics for summary statistics
        summary_metrics = {
            # Model selection parameters
            'best_m': ['mean', 'std'],
            'best_k': ['mean', 'std'],
            'best_k_bagged_gs': ['mean', 'std'],
            'best_k_smeared_gs': ['mean', 'std'],
            'best_k_gs': ['mean', 'std'],
            'best_noise_scale': ['mean', 'std'],
            
            # Lasso metrics
            'insample_lasso': ['mean', 'std'],
            'mse_lasso': ['mean', 'std'],
            'df_lasso': ['mean', 'std'],
            'coef_recovery_lasso': ['mean', 'std'],
            'support_recovery_lasso': ['mean', 'std'],
            'outsample_mse_lasso': ['mean', 'std'],
            'rte_lasso': ['mean', 'std'],
            'f_score_lasso': ['mean', 'std'],
            
            # Ridge metrics
            'insample_ridge': ['mean', 'std'],
            'mse_ridge': ['mean', 'std'],
            'df_ridge': ['mean', 'std'],
            'coef_recovery_ridge': ['mean', 'std'],
            'outsample_mse_ridge': ['mean', 'std'],
            'rte_ridge': ['mean', 'std'],
            'f_score_ridge': ['mean', 'std'],
            
            # ElasticNet metrics
            'insample_elastic': ['mean', 'std'],
            'mse_elastic': ['mean', 'std'],
            'df_elastic': ['mean', 'std'],
            'coef_recovery_elastic': ['mean', 'std'],
            'support_recovery_elastic': ['mean', 'std'],
            'outsample_mse_elastic': ['mean', 'std'],
            'rte_elastic': ['mean', 'std'],
            'f_score_elastic': ['mean', 'std'],
            
            # BaggedGS metrics
            'insample_bagged_gs': ['mean', 'std'],
            'mse_bagged_gs': ['mean', 'std'],
            'df_bagged_gs': ['mean', 'std'],
            'coef_recovery_bagged_gs': ['mean', 'std'],
            'support_recovery_bagged_gs': ['mean', 'std'],
            'outsample_mse_bagged_gs': ['mean', 'std'],
            'rte_bagged_gs': ['mean', 'std'],
            'f_score_bagged_gs': ['mean', 'std'],
            
            # SmearedGS metrics
            'insample_smeared_gs': ['mean', 'std'],
            'mse_smeared_gs': ['mean', 'std'],
            'df_smeared_gs': ['mean', 'std'],
            'coef_recovery_smeared_gs': ['mean', 'std'],
            'support_recovery_smeared_gs': ['mean', 'std'],
            'outsample_mse_smeared_gs': ['mean', 'std'],
            'rte_smeared_gs': ['mean', 'std'],
            'f_score_smeared_gs': ['mean', 'std'],
            
            # RGSCV metrics
            'insample_rgs': ['mean', 'std'],
            'mse_rgs': ['mean', 'std'],
            'df_rgs': ['mean', 'std'],
            'coef_recovery_rgs': ['mean', 'std'],
            'support_recovery_rgs': ['mean', 'std'],
            'outsample_mse_rgs': ['mean', 'std'],
            'rte_rgs': ['mean', 'std'],
            'f_score_rgs': ['mean', 'std'],
            
            # Greedy Selection metrics
            'insample_gs': ['mean', 'std'],
            'mse_gs': ['mean', 'std'],
            'df_gs': ['mean', 'std'],
            'coef_recovery_gs': ['mean', 'std'],
            'support_recovery_gs': ['mean', 'std'],
            'outsample_mse_gs': ['mean', 'std'],
            'rte_gs': ['mean', 'std'],
            'f_score_gs': ['mean', 'std']
        }
        
        # Filter metrics to only include those actually present in the data
        available_metrics = {}
        for metric, stats in summary_metrics.items():
            if metric in results_df.columns:
                available_metrics[metric] = stats
        
        # Create summary statistics
        summary = results_df.groupby('sigma').agg(available_metrics).round(4)
        
        return summary
    
    def _create_timing_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create timing summary statistics.
        
        Parameters
        ----------
        results_df : DataFrame
            Raw results containing timing information
            
        Returns
        -------
        DataFrame
            Timing summary statistics grouped by sigma
        """
        timing_metrics = {
            'time_lasso': ['mean', 'std', 'min', 'max'],
            'time_ridge': ['mean', 'std', 'min', 'max'],
            'time_elastic': ['mean', 'std', 'min', 'max'],
            'time_bagged_gs': ['mean', 'std', 'min', 'max'],
            'time_smeared_gs': ['mean', 'std', 'min', 'max'],
            'time_rgscv': ['mean', 'std', 'min', 'max'],
            'time_original_gs': ['mean', 'std', 'min', 'max']
        }
        
        # Filter timing metrics to only include those present in the data
        available_timing_metrics = {}
        for metric, stats in timing_metrics.items():
            if metric in results_df.columns:
                available_timing_metrics[metric] = stats
        
        # Create timing summary
        timing_summary = results_df.groupby('sigma').agg(available_timing_metrics).round(4)
        
        return timing_summary
    
    def _save_results(self, results_df: pd.DataFrame, summary: pd.DataFrame,
                     timing_summary: pd.DataFrame) -> str:
        """
        Save all results to files.
        
        Parameters
        ----------
        results_df : DataFrame
            Raw results from all experiments
        summary : DataFrame
            Summary statistics
        timing_summary : DataFrame
            Timing summary statistics
            
        Returns
        -------
        str
            Base filename used for saving
        """
        # Create filename base using descriptive parameters
        filename_base = (
            f"{self.params['data']['covariance_type']}_"
            f"{self.params['data']['generator_type']}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create save directory
        save_path = Path(self.params['output']['save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results and summaries
        results_df.to_csv(save_path / f'simulation_results_{filename_base}.csv', index=False)
        summary.to_csv(save_path / f'simulation_summary_{filename_base}.csv')
        timing_summary.to_csv(save_path / f'simulation_timing_summary_{filename_base}.csv')
        
        # Save parameters used
        with open(save_path / f'simulation_params_{filename_base}.json', 'w') as f:
            json.dump(self.params, f, indent=4)
        
        return filename_base
    
    def run_full_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete simulation pipeline.
        
        This is the main entry point that coordinates the entire simulation
        workflow from parameter loading to result saving.
        
        Returns
        -------
        tuple
            (results_df, summary_df, timing_summary_df)
        """
        print("Starting Simulation Pipeline")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Setup design matrix
        print("\n1. Setting up design matrix...")
        X, cov_matrix = self._setup_design_matrix()
        
        # Get sigma values
        print("\n2. Computing sigma values...")
        sigmas = self._get_sigma_values(X, cov_matrix)
        print(f"   Using {len(sigmas)} sigma values: {sigmas}")
        
        # Run experiment grid
        print("\n3. Running experiment grid...")
        print(f"   Total experiments: {self.params['simulation']['n_sim']} simulations × {len(sigmas)} sigma values = {self.params['simulation']['n_sim'] * len(sigmas)}")
        
        all_results = self._run_experiment_grid(X, cov_matrix, sigmas)
        
        # Convert results to DataFrame
        print("\n4. Processing results...")
        results_df = pd.DataFrame(all_results)
        
        # Create summary statistics
        print("5. Creating summary statistics...")
        summary = self._create_summary_statistics(results_df)
        timing_summary = self._create_timing_summary(results_df)
        
        # Save results
        print("6. Saving results...")
        filename_base = self._save_results(results_df, summary, timing_summary)
        
        # Print completion summary
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED")
        print("=" * 60)
        print(f"Total runtime: {total_time/60:.1f} minutes")
        print(f"Results saved with base filename: {filename_base}")
        print(f"Results shape: {results_df.shape}")
        print(f"Save location: {self.params['output']['save_path']}")
        
        # Print some key summary statistics
        print("\nKey Results Summary:")
        print("-" * 30)
        if 'rte_lasso' in summary.columns:
            print("RTE (Relative Test Error) Averages:")
            for model in ['lasso', 'bagged_gs', 'rgs']:
                rte_col = f'rte_{model}'
                if rte_col in summary.columns:
                    mean_rte = summary[rte_col]['mean'].mean()
                    print(f"  {model.upper():>10}: {mean_rte:.3f}")
        
        return results_df, summary, timing_summary
    
    @classmethod
    def run_from_config(cls, param_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convenience method to run simulation from a parameter file.
        
        Parameters
        ----------
        param_path : str
            Path to the simulation parameters JSON file
            
        Returns
        -------
        tuple
            (results_df, summary_df, timing_summary_df)
        """
        pipeline = cls(param_path)
        return pipeline.run_full_simulation() 