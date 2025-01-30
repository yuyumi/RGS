# def run_simulation(n_predictors=500, n_train=2000, signal_proportion=0.02, cov='orthogonal', n_sim=10):
#     """
#     Run simulation comparing different methods, varying k_max during training.
#     """
    
#     # Start timing
#     start_time = time.time()
    
#     # Generate base design matrix
#     X_generators = {
#         'orthogonal': generate_orthogonal_X,
#         'banded': generate_banded_X
#     }
#     if isinstance(cov, str):
#         if cov not in X_generators:
#             raise ValueError(f"Unknown generator: {cov}. Available generators: {list(X_generators.keys())}")
#         X_generator = X_generators[cov]
#     else:
#         # Assume it's a callable
#         X_generator = cov
#     X = X_generator(n_predictors, n_train)
    
#     # Define k values for training and m values for RGS
#     k_values = list(range(5, 31, 5))  # [5, 10, 15, ..., 50]
#     base = 2
#     num_points = 7
#     m_values = [int(2 + (n_predictors-2) * (base**x - 1)/(base**(num_points-1) - 1)) 
#             for x in range(num_points)]
    
#     # Initialize results storage
#     results = []
    
#     # Define example generators with different noise levels
#     sigmas = [0.01, 0.5, 1, 3, 5, 7]
# #     sigmas = [10, 20, 30, 35, 40, 45]
#     example_generators = {
# #         f'sigma_{sigmas[0]}': generate_exact_sparsity_example,
# #         f'sigma_{sigmas[1]}': generate_exact_sparsity_example,
# #         f'sigma_{sigmas[2]}': generate_exact_sparsity_example,
# #         f'sigma_{sigmas[3]}': generate_exact_sparsity_example,
# #         f'sigma_{sigmas[4]}': generate_exact_sparsity_example,
# #         f'sigma_{sigmas[5]}': generate_exact_sparsity_example
# #         f'sigma_{sigmas[0]}_inexact': generate_inexact_sparsity_example,
# #         f'sigma_{sigmas[1]}_inexact': generate_inexact_sparsity_example,
# #         f'sigma_{sigmas[2]}_inexact': generate_inexact_sparsity_example,
# #         f'sigma_{sigmas[3]}_inexact': generate_inexact_sparsity_example,
# #         f'sigma_{sigmas[4]}_inexact': generate_inexact_sparsity_example,
# #         f'sigma_{sigmas[5]}_inexact': generate_inexact_sparsity_example
# #         'sigma_10_linear': generate_nonlinear_example,
# #         'sigma_10_highly_nonlinear': generate_nonlinear_example,
# #         'sigma_60_small_snr': generate_small_snr,
# #         f'sigma_{sigmas[0]}_laplace': generate_laplace_example,
# #         f'sigma_{sigmas[1]}_laplace': generate_laplace_example,
# #         f'sigma_{sigmas[2]}_laplace': generate_laplace_example
# #         f'sigma_{sigmas[0]}_cauchy': generate_cauchy_example,
# #         f'sigma_{sigmas[1]}_cauchy': generate_cauchy_example,
# #         f'sigma_{sigmas[2]}_cauchy': generate_cauchy_example
#     }
    
#     # Create progress bar for total iterations
#     total_iterations = n_sim * len(example_generators)
#     pbar = tqdm(total=total_iterations, desc="Overall Progress")
    
#     # Time the first iteration separately to get a good estimate
#     first_iter_time = None
    
#     for sim in range(n_sim):
#         i = 0
#         for noise_level, generator in example_generators.items():
#             iter_start_time = time.time()
#             # Generate data for this simulation
# #             if (noise_level == 'sigma_10_linear'):
# #                 flag = 0
# #             else:
# #                 flag = 1
# #             X, y, y_true, p, sigma = generator(X, seed=sim, eta=flag)
#             X, y, y_true, beta_true, p, sigma = generator(X, signal_proportion, sigmas[i], seed=sim)
            
#             result = {
#                 'simulation': sim,
#                 'noise_level': noise_level,
#                 'sigma': sigma
#             }
            
#             # Fit Lasso
#             lasso = LassoCV(cv=10, random_state=sim)
#             lasso.fit(X, y)
#             y_pred_lasso = lasso.predict(X)
#             result['mse_lasso'] = mean_squared_error(y_true, y_pred_lasso)
#             result['df_lasso'] = calculate_df(y, y_true, y_pred_lasso, n_train, sigma)
#             result['coef_recovery_lasso'] = np.mean((lasso.coef_ - beta_true)**2)
#             result['support_recovery_lasso'] = np.mean((lasso.coef_ != 0) == (beta_true != 0))
            
#             # Fit Ridge
#             ridge = RidgeCV(cv=10)
#             ridge.fit(X, y)
#             y_pred_ridge = ridge.predict(X)
#             result['mse_ridge'] = mean_squared_error(y_true, y_pred_ridge)
#             result['df_ridge'] = calculate_df(y, y_true, y_pred_ridge, n_train, sigma)
#             result['coef_recovery_ridge'] = np.mean((ridge.coef_ - beta_true)**2)
            
#             # Fit Elastic Net
#             elastic = ElasticNetCV(cv=10, random_state=sim)
#             elastic.fit(X, y)
#             y_pred_elastic = elastic.predict(X)
#             result['mse_elastic'] = mean_squared_error(y_true, y_pred_elastic)
#             result['df_elastic'] = calculate_df(y, y_true, y_pred_elastic, n_train, sigma)
#             result['coef_recovery_elastic'] = np.mean((elastic.coef_ - beta_true)**2)
#             result['support_recovery_elastic'] = np.mean((elastic.coef_ != 0) == (beta_true != 0))
            
#             # Fit FGS
#             mse_fgs = {}
#             for k_max in k_values:
#                 fgs = RGS(k_max=k_max, m=n_predictors, n_resample_iter=7)
#                 fgs.fit(X, y)
#                 y_pred_fgs = fgs.predict(X, k=k_max)
#                 result[f'mse_gs_k{k_max}'] = mean_squared_error(y_true, y_pred_fgs)
#                 result[f'df_gs_k{k_max}'] = calculate_df(y, y_true, y_pred_fgs, n_train, sigma)
#                 result[f'pen_gs_k{k_max}'] = compute_penalized_score(fgs, X, y_true, k_max, sigma, n_train, p)
#                 result[f'coef_recovery_gs_k{k_max}'] = np.mean((fgs.coef_ - beta_true)**2)
#                 result[f'support_recovery_gs_k{k_max}'] = np.mean((fgs.coef_ != 0) == (beta_true != 0))
            
#             # Fit RGS
#             mse_rgs = {}
#             for m in m_values:
#                 for k_max in k_values:
#                     rgs = RGS(k_max=k_max, m=m, n_resample_iter=7)
#                     rgs.fit(X, y)
#                     y_pred_rgs = rgs.predict(X, k=k_max)
#                     result[f'mse_rgs_m{m}_k{k_max}'] = mean_squared_error(y_true, y_pred_rgs)
#                     result[f'df_rgs_m{m}_k{k_max}'] = calculate_df(y, y_true, y_pred_rgs, n_train, sigma)
#                     result[f'pen_rgs_m{m}_k{k_max}'] = compute_penalized_score(rgs, X, y_true, k_max, sigma, n_train, p)
#                     result[f'coef_recovery_rgs_m{m}_k{k_max}'] = np.mean((rgs.coef_ - beta_true)**2)
#                     result[f'support_recovery_rgs_m{m}_k{k_max}'] = np.mean((rgs.coef_ != 0) == (beta_true != 0))
            
#             results.append(result)
            
#             # Calculate timing information
#             iter_time = time.time() - iter_start_time
#             if first_iter_time is None:
#                 first_iter_time = iter_time
#                 estimated_total_time = first_iter_time * total_iterations
            
#             iterations_completed = sim * len(example_generators) + list(example_generators.keys()).index(noise_level) + 1
#             time_elapsed = time.time() - start_time
#             time_per_iter = time_elapsed / iterations_completed
#             estimated_remaining_time = time_per_iter * (total_iterations - iterations_completed)
            
#             # Update progress bar with timing information
#             pbar.set_postfix({
#                 'Simulation': f'{sim + 1}/{n_sim}',
#                 'Noise': noise_level,
#                 'Iter Time': f'{iter_time:.1f}s',
#                 'Est. Remaining': f'{estimated_remaining_time/60:.1f}min',
#                 'Est. Total': f'{(time_elapsed + estimated_remaining_time)/60:.1f}min'
#             })
#             pbar.update(1)
            
#             i += 1
            
#     # Close progress bar
#     pbar.close()
#     total_time = time.time() - start_time
#     print(f"\nSimulation completed in {total_time/60:.1f} minutes")
#     print(f"Average time per iteration: {total_time/total_iterations:.1f} seconds")
    
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Calculate summary statistics for both MSE and DF
#     metrics = ['mse', 'df']
#     base_methods = ['lasso', 'ridge', 'elastic']
#     agg_cols = {}
    
#     # Add base methods
#     for metric in metrics:
#         for method in base_methods:
#             col = f'{metric}_{method}'
#             agg_cols[col] = ['mean', 'std']
    
#     # Add FGS methods
#     for metric in metrics:
#         for k in k_values:
#             col = f'{metric}_gs_k{k}'
#             agg_cols[col] = ['mean', 'std']
    
#     # Add RGS methods
#     for metric in metrics:
#         for m in m_values:
#             for k in k_values:
#                 col = f'{metric}_rgs_m{m}_k{k}'
#                 agg_cols[col] = ['mean', 'std']
    
#     summary = results_df.groupby('noise_level').agg(agg_cols).round(4)
    
#     # Save results with timestamp
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     results_df.to_csv(f'../data/simulation_results_{timestamp}.csv', index=False)
#     summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
#     summary.to_csv(f'../data/simulation_summary_{timestamp}.csv')
    
#     print(f"\nResults saved to simulation_results_{timestamp}.csv")
#     print(f"Summary saved to simulation_summary_{timestamp}.csv")
    
#     return results_df, summary

# import numpy as np
# import pandas as pd
# import time
# import multiprocessing
# import psutil
# import warnings
# from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
# from sklearn.metrics import mean_squared_error
# from IPython.display import clear_output
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import shared_memory
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List, Tuple, Optional, Union
# import os
# from datetime import datetime


# class ProgressDisplay:
#     """Enhanced progress display with ETA and memory usage tracking"""
#     def __init__(self, total, desc="Progress"):
#         self.total = total
#         self.desc = desc
#         self.current = 0
#         self.start_time = time.time()
#         self._last_update = 0
#         self.update_interval = 0.1  # seconds
        
#     def update(self, n=1):
#         self.current += n
#         current_time = time.time()
#         if current_time - self._last_update >= self.update_interval:
#             self._display_progress()
#             self._last_update = current_time
        
#     def _display_progress(self):
#         clear_output(wait=True)
#         percentage = (self.current / self.total) * 100
#         elapsed_time = time.time() - self.start_time
#         rate = self.current / elapsed_time if elapsed_time > 0 else 0
#         eta = (self.total - self.current) / rate if rate > 0 else 0
        
#         # Memory usage (if psutil is available)
#         try:
#             import psutil
#             process = psutil.Process()
#             memory_usage = process.memory_info().rss / 1024 / 1024  # MB
#             memory_info = f", Memory: {memory_usage:.1f}MB"
#         except ImportError:
#             memory_info = ""
        
#         progress_bar = f"[{'=' * int(percentage/2)}{' ' * (50-int(percentage/2))}]"
#         print(f"{self.desc}: {progress_bar} {percentage:.1f}%")
#         print(f"Progress: {self.current}/{self.total}")
#         print(f"Elapsed: {elapsed_time:.1f}s, ETA: {eta:.1f}s, Rate: {rate:.1f} it/s{memory_info}")

# def _run_single_simulation(args):
#     """Helper function for parallel processing"""
#     (sim, noise_level, generator, X_shape, X_dtype, shm_name, 
#      m_values, k_max, k_values, n_train, signal_proportion, sigmas) = args
    
#     try:
#         # Get X from shared memory
#         existing_shm = shared_memory.SharedMemory(name=shm_name)
#         X = np.ndarray(X_shape, dtype=X_dtype, buffer=existing_shm.buf)
        
#         # Rest of your original function
#         X, y, y_true, beta_true, p, sigma = generator(X, signal_proportion, 
#                                                      sigmas[noise_level], seed=sim)
        
#         # Fit models and compute metrics
#         results = []
        
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore')
            
#             # Baseline models
#             models = {
#                 'lasso': LassoCV(cv=5, random_state=sim),
#                 'ridge': RidgeCV(cv=5),
#                 'elastic': ElasticNetCV(cv=5, random_state=sim)
#             }
            
#             base_metrics = {}
#             for name, model in models.items():
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#                 base_metrics.update({
#                     f'mse_{name}': mean_squared_error(y_true, y_pred),
#                     f'df_{name}': calculate_df(y, y_true, y_pred, n_train, sigma),
#                     f'coef_recovery_{name}': np.mean((model.coef_ - beta_true)**2),
#                     f'support_recovery_{name}': np.mean((np.abs(model.coef_) > 1e-10) == (beta_true != 0))
#                 })
            
#             # GS model
#             gs = RGS(k_max=k_max, m=n_predictors, n_resample_iter=7)
#             gs.fit(X, y)
#             for k in k_values:
#                 y_pred_gs = gs.predict(X, k=k)
#                 base_metrics.update({
#                     f'mse_gs_k{k}': mean_squared_error(y_true, y_pred_gs),
#                     f'df_gs_k{k}': calculate_df(y, y_true, y_pred_gs, n_train, sigma),
#                     f'pen_gs_k{k}': compute_penalized_score(gs, X, y_true, k, sigma, n_train, p),
#                     f'coef_recovery_gs_k{k}': np.mean((gs.coef_[k] - beta_true)**2),
#                     f'support_recovery_gs_k{k}': np.mean((np.abs(gs.coef_[k]) > 1e-10) == (beta_true != 0))
#                 })
            
#             # RGS models
#             for m in m_values:
#                 rgs = RGS(k_max=k_max, m=m, n_resample_iter=7)
#                 rgs.fit(X, y)
                
#                 for k in range(1, k_max + 1):
#                     y_pred = rgs.predict(X, k=k)
#                     result = {
#                         'simulation': sim,
#                         'noise_level': f'sigma_{sigmas[noise_level]}',
#                         'm': m,
#                         'k': k,
#                         'sigma': sigma,
#                         **base_metrics,
#                         f'mse_rgs_m{m}_k{k}': mean_squared_error(y_true, y_pred),
#                         f'df_rgs_m{m}_k{k}': calculate_df(y, y_true, y_pred, n_train, sigma),
#                         f'pen_rgs_m{m}_k{k}': compute_penalized_score(rgs, X, y_true, k, sigma, n_train, p),
#                         f'coef_recovery_rgs_m{m}_k{k}': np.mean((rgs.coef_[k] - beta_true)**2),
#                         f'support_recovery_rgs_m{m}_k{k}': np.mean((np.abs(rgs.coef_[k]) > 1e-10) == (beta_true != 0))
#                     }
#                     results.append(result)
                    
#         return results
    
#     except Exception as e:
#         print(f"Error in simulation {sim}, noise_level {noise_level}: {str(e)}")
#         return []
#     finally:
#         # Clean up shared memory in worker
#         existing_shm.close()

# def run_simulation(n_predictors=500, n_train=2000, signal_proportion=0.02, 
#                   cov='orthogonal', n_sim=10, n_jobs=-1):
#     """Run simulation with optimal CPU utilization.
    
#     Parameters
#     ----------
#     n_predictors : int, default=500
#         Number of predictor variables
#     n_train : int, default=2000
#         Number of training samples
#     signal_proportion : float, default=0.02
#         Proportion of relevant features
#     cov : str or callable, default='orthogonal'
#         Covariance structure ('orthogonal' or 'banded') or custom generator
#     n_sim : int, default=10
#         Number of simulation repetitions
#     n_jobs : int, optional
#         Number of parallel jobs. If None, automatically determined.
#         If -1, uses all CPU cores except one.
#     """
#     # Determine optimal number of workers
#     if n_jobs is None:
#         n_jobs = get_optimal_workers()
#     elif n_jobs == -1:
#         n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
#     print(f"Running simulation with {n_jobs} parallel workers")
#     """Run simulation finding optimal (m,k) pairs with parallel processing.
    
#     Parameters
#     ----------
#     n_predictors : int, default=500
#         Number of predictor variables
#     n_train : int, default=2000
#         Number of training samples
#     signal_proportion : float, default=0.02
#         Proportion of relevant features
#     cov : str or callable, default='orthogonal'
#         Covariance structure ('orthogonal' or 'banded') or custom generator
#     n_sim : int, default=10
#         Number of simulation repetitions
#     n_jobs : int, default=-1
#         Number of parallel jobs (-1 for all cores)
    
#     Returns
#     -------
#     tuple
#         (results_df, summary_df) containing detailed and summarized results
#     """
#     start_time = time.time()
    
#     # Generate base design matrix
#     X_generators = {
#         'orthogonal': generate_orthogonal_X,
#         'banded': generate_banded_X
#     }
#     X_generator = X_generators[cov] if isinstance(cov, str) else cov
#     X = X_generator(n_predictors, n_train)

#     # Create shared memory for X
#     shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
#     # Create a NumPy array backed by shared memory
#     X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
#     # Copy the data
#     X_shared[:] = X[:]

#     # Define noise levels
#     sigma_dict = {
#         'orthogonal': [1, 3, 5, 10, 15],
#         'banded': [0.2, 1, 3, 5, 10]
#     }
#     sigmas = sigma_dict[cov]
#     example_generators = {i: generate_laplace_example for i in range(len(sigmas))}
    
#     # Define parameter grids
#     k_values = [1, 4, 7, 10, 15, 20, 25]
#     base = 2
#     num_points = 7
#     m_values = [int(2 + (n_predictors-2) * (base**x - 1)/(base**(num_points-1) - 1)) 
#                 for x in range(num_points)]
#     k_max = 35
    
#     # Create shared memory for X
#     shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
#     X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
#     X_shared[:] = X[:]

#     try:
#         # Modify args_list to pass the shared memory name
#         args_list = [(sim, noise_level, generator, X_shared.shape, X_shared.dtype, 
#                      shm.name, m_values, k_max, k_values, n_train, 
#                      signal_proportion, sigmas)
#                     for sim in range(n_sim)
#                     for noise_level, generator in example_generators.items()]
        
#         # Run simulations in parallel
#         all_results = []
#         with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#             for results in executor.map(_run_single_simulation, args_list):
#                 all_results.extend(results)
#                 progress.update()
        
#         # Convert to DataFrame and compute summary
#         results_df = pd.DataFrame(all_results)
#         summary = compute_summary_statistics(results_df)
        
#         # Save results
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         results_df.to_csv(f'simulation_results_{timestamp}.csv', index=False)
#         summary.to_csv(f'simulation_summary_{timestamp}.csv')
        
#         print(f"\nSimulation completed in {(time.time() - start_time)/60:.1f} minutes")
        
#         return results_df, summary
        
#     finally:
#         # Clean up shared memory
#         shm.close()
#         shm.unlink()

# def compute_summary_statistics(results_df):
#     """Compute summary statistics for simulation results."""
#     # Basic metrics (lasso, ridge, elastic)
#     basic_metrics = ['mse', 'df', 'coef_recovery', 'support_recovery']
#     basic_methods = ['lasso', 'ridge', 'elastic']
    
#     summary_metrics = {}
    
#     # Basic methods
#     for method in basic_methods:
#         for metric in basic_metrics:
#             col = f'{metric}_{method}'
#             if col in results_df.columns:
#                 summary_metrics[col] = ['mean', 'std', 'min', 'max']
    
#     # RGS metrics
#     m_values = results_df['m'].unique()
#     k_values = results_df['k'].unique()
    
#     for m in m_values:
#         for k in k_values:
#             metrics = ['mse', 'df', 'pen', 'coef_recovery', 'support_recovery']
#             for metric in metrics:
#                 col = f'{metric}_rgs_m{m}_k{k}'
#                 if col in results_df.columns:
#                     summary_metrics[col] = ['mean', 'std', 'min', 'max']
    
#     # GS metrics
#     for k in k_values:
#         metrics = ['mse', 'df', 'pen', 'coef_recovery', 'support_recovery']
#         for metric in metrics:
#             col = f'{metric}_gs_k{k}'
#             if col in results_df.columns:
#                 summary_metrics[col] = ['mean', 'std', 'min', 'max']
    
#     return results_df.groupby(['noise_level', 'm', 'k']).agg(summary_metrics).round(4)

# def plot_optimal_mk(results_df, metric='mse_rgs', title=None):
#     """Enhanced plotting function for optimal (m,k) pairs.
    
#     Parameters
#     ----------
#     results_df : pd.DataFrame
#         Simulation results
#     metric : str, default='mse_rgs'
#         Metric to optimize ('mse_rgs', 'pen_rgs', etc.)
#     title : str, optional
#         Custom plot title
    
#     Returns
#     -------
#     matplotlib.figure.Figure
#         Plot figure
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     plt.figure(figsize=(12, 6))
#     sns.set_style("whitegrid")
    
#     # Find optimal k for each m and noise level
#     optimal_k = results_df.groupby(['noise_level', 'm'])[metric].idxmin()
#     optimal_params = results_df.loc[optimal_k, ['noise_level', 'm', 'k']]
    
#     # Plot with error bars
#     for noise_level in optimal_params['noise_level'].unique():
#         data = optimal_params[optimal_params['noise_level'] == noise_level]
#         mean_k = data.groupby('m')['k'].mean()
#         std_k = data.groupby('m')['k'].std()
        
#         plt.errorbar(mean_k.index, mean_k.values, yerr=std_k.values,
#                     label=noise_level, marker='o', capsize=5)
    
#     plt.xlabel('Number of Candidates (m)')
#     plt.ylabel('Optimal k')
#     plt.title(title or f'Optimal k vs m for Different Noise Levels\nOptimized for {metric}')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     return plt.gcf()