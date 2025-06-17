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


def _run_single_simulation_worker(args):
    """
    Standalone worker function for parallel execution.
    Must be at module level to be picklable.
    """
    sim_num, params = args
    
    try:
        # Create fresh instances for this process
        from rgs_experiments.simulation.data.data_generator import DataGenerator
        from rgs_experiments.simulation.orchestration.experiment_orchestrator import ExperimentOrchestrator
        from rgs_experiments.simulation.config.parameter_loader import get_sigma_list
        from tqdm import tqdm
        
        # Initialize components for this process
        data_generator = DataGenerator(params)
        experiment_orchestrator = ExperimentOrchestrator(params)
        
        base_seed = params['simulation']['base_seed']
        sim_results = []
        
        # Generate X for this simulation
        X_sim, cov_matrix = data_generator.generate_design_matrix(
            seed=base_seed + sim_num * 10000
        )
        
        # Get sigma values based on this X
        sigma_config = params['simulation']['sigma']
        sigmas = get_sigma_list(
            sigma_config,
            target_covariance=cov_matrix,
            signal_proportion=params['data']['signal_proportion'],
            generator_type=params['data']['generator_type'],
            eta=params['data']['generator_params'].get('eta', 0.5),
            seed=base_seed
        )
        
        # Verbose output for this simulation
        print(f"[Sim {sim_num}] Running {len(sigmas)} sigma values")
        
        # Run all sigma values with this same X with progress bar
        with tqdm(total=len(sigmas), desc=f"Sim {sim_num}", position=sim_num, leave=False) as pbar:
            for i, sigma in enumerate(sigmas, 1):
                # Update progress bar with current sigma
                pbar.set_description(f"Sim {sim_num} σ={sigma:.3f}")
                
                result = experiment_orchestrator.run_single_experiment(
                    X=X_sim,
                    cov_matrix=cov_matrix,
                    sigma=sigma,
                    seed=base_seed,
                    sim_num=sim_num
                )
                sim_results.append(result)
                pbar.update(1)
        
        print(f"[Sim {sim_num}] Completed all {len(sigmas)} sigma values")
        return sim_results
        
    except Exception as e:
        print(f"Error in simulation {sim_num}: {e}")
        import traceback
        traceback.print_exc()
        return []


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
        
        # Determine design type for logging
        if self.params['data']['covariance_type'] == 'block':
            fixed_design = self.params['data']['block_params'].get('fixed_design', False)
        elif self.params['data']['covariance_type'] == 'banded':
            fixed_design = self.params['data']['banded_params'].get('fixed_design', False)
        else:
            raise ValueError(f"Unknown covariance type: {self.params['data']['covariance_type']}")
        
        # Check matrix rank and log information
        rank_info = check_matrix_rank(X)
        
        print(f"Design Matrix Information:")
        print(f"  - Covariance type: {self.params['data']['covariance_type']}")
        print(f"  - Design type: {'Fixed' if fixed_design else 'Random'}")
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
            fixed_design = self.params['data']['block_params'].get('fixed_design', False)
        elif self.params['data']['covariance_type'] == 'banded':
            fixed_design = self.params['data']['banded_params'].get('fixed_design', False)
        else:
            raise ValueError(f"Unknown covariance type: {self.params['data']['covariance_type']}")
        
        sigma_config = self.params['simulation']['sigma']
        sigmas = get_sigma_list(
            sigma_config,
            target_covariance=cov_matrix,
            signal_proportion=self.params['data']['signal_proportion'],
            generator_type=self.params['data']['generator_type'],
            eta=self.params['data']['generator_params'].get('eta', 0.5),
            seed=self.params['simulation']['base_seed']
        )
        
        # If PVE was used, also store the original PVE values for later use
        if sigma_config.get('type') == 'pve':
            if sigma_config['style'] == 'list':
                pve_values = sigma_config['values']
            elif sigma_config['style'] == 'range':
                pve_values = np.linspace(
                    sigma_config['params']['min'],
                    sigma_config['params']['max'],
                    sigma_config['params']['num_points']
                )
            else:
                pve_values = []
            
            # Create sigma-to-PVE mapping for later lookup
            sigma_pve_pairs = list(zip(sigmas, pve_values))
            # Sort by sigma to maintain order
            sigma_pve_pairs = sorted(sigma_pve_pairs, key=lambda x: x[0])
            sigmas = [pair[0] for pair in sigma_pve_pairs]
            pve_values = [pair[1] for pair in sigma_pve_pairs]
            
            # Store both sigma and PVE values
            self.params['simulation']['sigma']['computed_values'] = sigmas
            self.params['simulation']['sigma']['pve_values'] = pve_values
        else:
            sigmas = sorted(sigmas)  # Sort for nice progression in progress bar
            # Save the actual sigma values used
            self.params['simulation']['sigma']['computed_values'] = sigmas
        
        return sigmas
    
    def _run_over_sigmas(self, sim_num: int) -> List[Dict[str, Any]]:
        """
        Run over all sigma values for a single simulation: generate X, compute sigmas, run all sigma values.
        
        Parameters
        ----------
        sim_num : int
            Simulation number
            
        Returns
        -------
        list
            List of results for all sigma values with the same design matrix X
        """
        base_seed = self.params['simulation']['base_seed']
        sim_results = []
        
        # Generate X for this simulation
        X_sim, cov_matrix = self.data_generator.generate_design_matrix(
            seed=base_seed + sim_num * 10000
        )
        
        # Get sigma values based on this X
        sigmas = self._get_sigma_values(X_sim, cov_matrix)
        
        # Verbose output for this simulation
        print(f"[Sim {sim_num}] Running {len(sigmas)} sigma values")
        
        # Run all sigma values with this same X with progress bar
        with tqdm(total=len(sigmas), desc=f"Sim {sim_num}", leave=False) as pbar:
            for i, sigma in enumerate(sigmas, 1):
                # Update progress bar with current sigma
                pbar.set_description(f"Sim {sim_num} σ={sigma:.3f}")
                
                # Run single experiment with this simulation's X
                result = self.experiment_orchestrator.run_single_experiment(
                    X=X_sim,
                    cov_matrix=cov_matrix,
                    sigma=sigma,
                    seed=base_seed,
                    sim_num=sim_num
                )
                
                sim_results.append(result)
                pbar.update(1)
        
        print(f"[Sim {sim_num}] Completed all {len(sigmas)} sigma values")
        return sim_results

    def _run_simulations_parallel(self, n_workers: int = None) -> List[Dict[str, Any]]:
        """
        Run all simulations in parallel.
        
        Parameters
        ----------
        n_workers : int, optional
            Number of parallel workers. If None, uses all available CPUs.
            
        Returns
        -------
        list
            List of all experiment results
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import cpu_count
        
        if n_workers is None:
            n_workers = min(cpu_count(), self.params['simulation']['n_sim'])
        
        print(f"   Using {n_workers} parallel workers")
        
        # Create arguments for each simulation
        sim_args = [
            (sim_num, self.params) 
            for sim_num in range(1, self.params['simulation']['n_sim'] + 1)
        ]
        
        # Execute in parallel with progress tracking
        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs using the standalone worker function
            future_to_sim = {
                executor.submit(_run_single_simulation_worker, args): args[0] 
                for args in sim_args
            }
            
            # Collect results with progress bar
            with tqdm(total=len(sim_args), desc="Parallel Simulations") as pbar:
                for future in as_completed(future_to_sim):
                    sim_num = future_to_sim[future]
                    try:
                        sim_results = future.result()
                        all_results.extend(sim_results)
                        
                        # Update progress bar
                        pbar.set_description(f"Completed simulation {sim_num}")
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error retrieving results for simulation {sim_num}: {e}")
                        pbar.update(1)
        
        return all_results

    def _run_simulations_sequential(self) -> List[Dict[str, Any]]:
        """
        Run all simulations sequentially.
        
        Returns
        -------
        list
            List of all experiment results
        """
        all_results = []
        
        # Run simulations with progress bar
        with tqdm(total=self.params['simulation']['n_sim'], desc="Sequential Simulations") as pbar:
            for sim in range(1, self.params['simulation']['n_sim'] + 1):
                # Update progress bar description
                pbar.set_description(f"Simulation {sim}/{self.params['simulation']['n_sim']}")
                
                # Run over all sigma values for this simulation
                sim_results = self._run_over_sigmas(sim)
                all_results.extend(sim_results)
                
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
        
        # Print simulation configuration
        print(f"Configuration:")
        print(f"  - Covariance type: {self.params['data']['covariance_type']}")
        print(f"  - Generator type: {self.params['data']['generator_type']}")
        if self.params['data']['covariance_type'] == 'block':
            fixed_design = self.params['data']['block_params'].get('fixed_design', False)
        elif self.params['data']['covariance_type'] == 'banded':
            fixed_design = self.params['data']['banded_params'].get('fixed_design', False)
        else:
            raise ValueError(f"Unknown covariance type: {self.params['data']['covariance_type']}")
        print(f"  - Design type: {'Fixed' if fixed_design else 'Random'}")
        print(f"  - Simulations: {self.params['simulation']['n_sim']}")
        
        self.start_time = time.time()
        
        # Determine execution mode
        parallel_config = self.params.get('execution', {})
        use_parallel = parallel_config.get('parallel', False)
        n_workers = parallel_config.get('n_workers', None)
        
        # Main simulation loop
        print("\n1. Running simulations...")
        print("   Each simulation generates its own design matrix X and sigma values")
        
        if use_parallel:
            print("   Using parallel execution")
            all_results = self._run_simulations_parallel(n_workers=n_workers)
        else:
            print("   Using sequential execution")
            all_results = self._run_simulations_sequential()
        
        print(f"   Total experiments completed: {len(all_results)}")
        
        # Check if we have any results
        if len(all_results) == 0:
            raise ValueError("No experiments completed successfully. Check error messages above.")
        
        # Convert results to DataFrame
        print("\n2. Processing results...")
        results_df = pd.DataFrame(all_results)
        
        # Create summary statistics
        print("3. Creating summary statistics...")
        summary = self._create_summary_statistics(results_df)
        timing_summary = self._create_timing_summary(results_df)
        
        # Save results
        print("4. Saving results...")
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