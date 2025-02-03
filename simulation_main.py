import numpy as np
import pandas as pd
import time
import json  # Add this import
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from functools import partial

from RGS.core.rgs import RGS, RGSCV
from RGS.penalized_score import create_penalized_scorer
from RGS.utils.sim_util_dgs import *

def load_params(param_path):
    """Load parameters from JSON file."""
    with open(param_path, 'r') as f:
        params = json.load(f)
    return params

def pve_to_sigma(pve, signal_proportion, n_predictors):
    """
    Convert PVE to sigma value.
    
    PVE = (s*p)/(s*p + sigma^2)
    where s is signal_proportion and p is n_predictors
    
    Solving for sigma:
    sigma = sqrt((s*p/PVE) - s*p)
    """
    sp = signal_proportion * n_predictors
    return np.sqrt((sp/pve) - sp)

def get_sigma_list(sigma_params, signal_proportion, n_predictors):
    """
    Get list of sigma values based on parameters.
    
    Parameters
    ----------
    sigma_params : dict
        Dictionary containing either:
        - type: "list" and values with list of sigma values
        - type: "pve" and params with num_points, min_pve, max_pve
    signal_proportion : float
        Signal proportion (needed for PVE calculation)
    n_predictors : int
        Number of predictors (needed for PVE calculation)
        
    Returns
    -------
    list
        List of sigma values to use
    """
    if sigma_params['type'] == 'list':
        return sigma_params['values']
    elif sigma_params['type'] == 'pve':
        pve_values = np.linspace(
            sigma_params['params']['min_pve'],
            sigma_params['params']['max_pve'],
            sigma_params['params']['num_points']
        )
        return [pve_to_sigma(pve, signal_proportion, n_predictors) 
                for pve in pve_values]
    else:
        raise ValueError(f"Unknown sigma type: {sigma_params['type']}")

def get_m_grid(grid_params, n_predictors):
    """
    Get m_grid based on parameters.
    
    Parameters
    ----------
    grid_params : dict
        Dictionary containing either:
        - type: "geometric" and params with base and num_points
        - type: "list" and values with list of m values
    n_predictors : int
        Number of predictors (needed for geometric grid)
    
    Returns
    -------
    list
        List of m values to use
    """
    if grid_params['type'] == 'geometric':
        base = grid_params['params']['base']
        num_points = grid_params['params']['num_points']
        return [int(2 + (n_predictors-2) * (base**x - 1)/(base**(num_points-1) - 1)) 
                for x in range(num_points)]
    elif grid_params['type'] == 'list':
        return grid_params['values']
    else:
        raise ValueError(f"Unknown m_grid type: {grid_params['type']}")

def run_one_dgp_iter(
    X,
    generator,
    sigma,
    params,
    seed
):
    """Run one iteration of the simulation for a specific DGP setting."""
    # Generate data
    X, y, y_true, beta_true, p, sigma = generator(
        X, 
        params['data']['signal_proportion'], 
        sigma, 
        seed=seed
    )
    n_train = X.shape[0]
    
    # Fit baseline models with specified CV
    cv = params['model']['baseline']['cv']
    
    # Lasso
    lasso = LassoCV(cv=cv, random_state=seed)
    lasso.fit(X, y)
    y_pred_lasso = lasso.predict(X)
    
    # Ridge
    ridge = RidgeCV(cv=cv)
    ridge.fit(X, y)
    y_pred_ridge = ridge.predict(X)
    
    # Elastic Net
    elastic = ElasticNetCV(cv=cv, random_state=seed)
    elastic.fit(X, y)
    y_pred_elastic = elastic.predict(X)
    
    # Create result dictionary with baseline models
    result = {
        'simulation': seed,
        'sigma': sigma,
        'mse_lasso': mean_squared_error(y_true, y_pred_lasso),
        'coef_recovery_lasso': np.mean((lasso.coef_ - beta_true)**2),
        'support_recovery_lasso': np.mean((lasso.coef_ != 0) == (beta_true != 0)),
        'mse_ridge': mean_squared_error(y_true, y_pred_ridge),
        'coef_recovery_ridge': np.mean((ridge.coef_ - beta_true)**2),
        'mse_elastic': mean_squared_error(y_true, y_pred_elastic),
        'coef_recovery_elastic': np.mean((elastic.coef_ - beta_true)**2),
        'support_recovery_elastic': np.mean((elastic.coef_ != 0) == (beta_true != 0))
    }
    
    ## Create penalized scorer
    scorer = create_penalized_scorer(sigma**2, n_train, p, params['model']['k_max'])
    
    # Calculate m_grid
    m_grid = get_m_grid(
        params['model']['m_grid'],
        params['data']['n_predictors']
    )
    
    # Fit RGSCV
    rgscv = RGSCV(
        k_max=params['model']['k_max'],
        m_grid=m_grid,
        n_replications=params['model']['rgscv']['n_replications'],
        n_resample_iter=params['model']['rgscv']['n_resample_iter'],
        random_state=seed,
        cv=params['model']['rgscv']['cv'],
        scoring=scorer
    )
    rgscv.fit(X, y)
    
    # Get predictions using best parameters
    y_pred_rgs = rgscv.predict(X)
    
    # Add RGSCV results
    result.update({
        'best_m': rgscv.m_,
        'best_k': rgscv.k_,
        'mse_rgs': mean_squared_error(y_true, y_pred_rgs),
        'coef_recovery_rgs': np.mean((rgscv.model_.coef_[rgscv.k_] - beta_true)**2),
        'support_recovery_rgs': np.mean(
            (np.abs(rgscv.model_.coef_[rgscv.k_]) > 1e-10) == (beta_true != 0)
        )
    })
    
    return result

def main(param_path):
    """
    Main simulation loop.
    """
    # Load parameters
    params = load_params(param_path)
    start_time = time.time()
    
    # Generate base design matrix
    X_generators = {
        'orthogonal': generate_orthogonal_X,
        'banded': generate_banded_X,
        'block': lambda n_predictors, n_train: generate_block_X(
            n_predictors=n_predictors,
            n_train=n_train,
            block_size=params['data']['block_params']['block_size'],
            within_correlation=params['data']['block_params']['within_correlation']
        )
    }
    
    X = X_generators[params['data']['covariance_type']](
        params['data']['n_predictors'],
        params['data']['n_train']
    )
    
    # Set up generator mapping
    generators = {
        'exact': generate_exact_sparsity_example,
        'inexact': generate_inexact_sparsity_example,
        'nonlinear': generate_nonlinear_example,
        'laplace': generate_laplace_example,
        'cauchy': generate_cauchy_example
    }
    generator = generators[params['data']['generator_type']]
    
    # Get sigma values based on parameters
    sigmas = get_sigma_list(
        params['simulation']['sigma'],
        params['data']['signal_proportion'],
        params['data']['n_predictors']
    )
    
    # Sort sigmas for nice progression in progress bar
    sigmas = sorted(sigmas)
    
    # Save the actual sigma values used in the params for reference
    params['simulation']['sigma']['computed_values'] = sigmas
    
    # Initialize results storage
    all_results = []
    
    # Main simulation loop with nested tqdm
    total_sims = params['simulation']['n_sim'] * len(sigmas)
    with tqdm(total=total_sims, desc="Total Progress") as pbar:
        for sim in range(params['simulation']['n_sim']):
            seed = params['simulation']['base_seed'] + sim
            
            for sigma in sigmas:
                pbar.set_description(
                    f"Sim {sim+1}/{params['simulation']['n_sim']}, "
                    f"σ={sigma}, {params['data']['generator_type']}, "
                    f"{params['data']['covariance_type']}"
                )
                
                result = run_one_dgp_iter(
                    X=X,
                    generator=generator,
                    sigma=sigma,
                    params=params,
                    seed=seed
                )
                all_results.append(result)
                pbar.update(1)
    
    # Convert results to DataFrame and compute summary
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    summary_metrics = {
        'best_m': ['mean', 'std'],
        'best_k': ['mean', 'std'],
        'mse_lasso': ['mean', 'std'],
        'coef_recovery_lasso': ['mean', 'std'],
        'support_recovery_lasso': ['mean', 'std'],
        'mse_ridge': ['mean', 'std'],
        'coef_recovery_ridge': ['mean', 'std'],
        'mse_elastic': ['mean', 'std'],
        'coef_recovery_elastic': ['mean', 'std'],
        'support_recovery_elastic': ['mean', 'std'],
        'mse_rgs': ['mean', 'std'],
        'coef_recovery_rgs': ['mean', 'std'],
        'support_recovery_rgs': ['mean', 'std']
    }
    
    summary = results_df.groupby('sigma').agg(summary_metrics).round(4)
    
   # Create filename base using descriptive parameters
    filename_base = (
        f"{params['data']['covariance_type']}_"
        f"{params['data']['generator_type']}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # Create save directory
    save_path = Path(params['output']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)

    # Save results and summary
    results_df.to_csv(save_path / f'simulation_results_{filename_base}.csv', index=False)
    summary.to_csv(save_path / f'simulation_summary_{filename_base}.csv')

    # Save parameters used
    with open(save_path / f'simulation_params_{filename_base}.json', 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\nSimulation completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved with base filename: {filename_base}")
        
    return results_df, summary

if __name__ == "__main__":
    # Get the project root directory (where setup.py is)
    root_dir = Path(__file__).parent
    param_path = root_dir / "params" / "sim_params.json"
    results_df, summary = main(param_path)