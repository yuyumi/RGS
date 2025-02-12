import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# Import from core RGS package
from rgs.core.rgs import RGSCV
from rgs.penalized_score import create_penalized_scorer

# Import from simulation package
from rgs_experiments.utils.sim_util_dgs import *
from rgs_experiments.models.data_smearing import DataSmearingRegressor

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
    seed,
    sim_num
):
    """Run one iteration of the simulation for a specific DGP setting."""
    # Get generator-specific parameters if needed
    if params['data']['generator_type'] in ['nonlinear', 'inexact']:
        generator_params = params['data'].get('generator_params', {})
        eta = generator_params.get('eta', 0.5)  # default to 0.5 if not specified
        X, y, y_true, beta_true, p, sigma = generator(
            X, 
            params['data']['signal_proportion'], 
            sigma,
            eta=eta,
            seed=seed+sim_num
        )
    else:
        # Original code for other generators
        X, y, y_true, beta_true, p, sigma = generator(
            X, 
            params['data']['signal_proportion'], 
            sigma, 
            seed=seed+sim_num
        )
    
    # Fit baseline models with specified CV
    cv = params['model']['baseline']['cv']
    
    # Initialize result dictionary
    result = {
        'simulation': sim_num,
        'sigma': sigma
    }
    
    # Fit and evaluate Lasso
    lasso = LassoCV(cv=cv, random_state=seed+sim_num)
    lasso.fit(X, y)
    y_pred_lasso = lasso.predict(X)
    result.update({
        'insample_lasso': mean_squared_error(y_true, y_pred_lasso),
        'mse_lasso': mean_squared_error(y, y_pred_lasso),
        'coef_recovery_lasso': np.mean((lasso.coef_ - beta_true)**2),
        'support_recovery_lasso': np.mean((lasso.coef_ != 0) == (beta_true != 0))
    })
    
    # Fit and evaluate Ridge
    ridge = RidgeCV(cv=cv)
    ridge.fit(X, y)
    y_pred_ridge = ridge.predict(X)
    result.update({
        'insample_ridge': mean_squared_error(y_true, y_pred_ridge),
        'mse_ridge': mean_squared_error(y, y_pred_ridge),
        'coef_recovery_ridge': np.mean((ridge.coef_ - beta_true)**2)
    })
    
    # Fit and evaluate Elastic Net
    elastic = ElasticNetCV(cv=cv, random_state=seed+sim_num)
    elastic.fit(X, y)
    y_pred_elastic = elastic.predict(X)
    result.update({
        'insample_elastic': mean_squared_error(y_true, y_pred_elastic),
        'mse_elastic': mean_squared_error(y, y_pred_elastic),
        'coef_recovery_elastic': np.mean((elastic.coef_ - beta_true)**2),
        'support_recovery_elastic': np.mean((elastic.coef_ != 0) == (beta_true != 0))
    })

    # Fit and evaluate Bagged Linear Regression
    param_grid = {
        'max_samples': params['model']['bagging']['param_grid']['max_samples'],
        'max_features': params['model']['bagging']['param_grid']['max_features']
    }
    
    bagging = BaggingRegressor(
        estimator=LinearRegression(),
        n_estimators=params['model']['bagging']['n_estimators'],
        random_state=seed+sim_num
    )
    
    grid_search = GridSearchCV(
        bagging,
        param_grid,
        cv=params['model']['bagging']['cv'],
        scoring='neg_mean_squared_error',
        refit=True
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    bagged_lr = BaggingRegressor(
        estimator=LinearRegression(),
        n_estimators=params['model']['bagging']['n_estimators'],
        max_samples=best_params['max_samples'],
        max_features=best_params['max_features'],
        random_state=seed+sim_num
    )
    bagged_lr.fit(X, y)
    
    y_pred_bagged = bagged_lr.predict(X)
    result.update({
        'insample_bagged': mean_squared_error(y_true, y_pred_bagged),
        'mse_bagged': mean_squared_error(y, y_pred_bagged),
        'coef_recovery_bagged': np.mean((
            np.mean([est.coef_ for est in bagged_lr.estimators_], axis=0) - beta_true
        )**2),
        'max_samples_bagged': bagged_lr.max_samples,
        'max_features_bagged': bagged_lr.max_features
    })

    # Fit and evaluate Data Smearing
    param_grid = {
        'noise_sigma': params['model']['smearing']['param_grid']['noise_sigma']
    }
    
    smearing = DataSmearingRegressor(
        n_estimators=params['model']['smearing']['n_estimators'],
        random_state=seed+sim_num
    )
    
    grid_search = GridSearchCV(
        smearing,
        param_grid,
        cv=params['model']['baseline']['cv'],
        scoring='neg_mean_squared_error',
        refit=True  # This ensures we refit on the full dataset
    )
    grid_search.fit(X, y)
    
    # Get the best parameters and refit on full dataset
    best_sigma = grid_search.best_params_['noise_sigma']
    smearing_reg = DataSmearingRegressor(
        n_estimators=params['model']['smearing']['n_estimators'],
        noise_sigma=best_sigma,
        random_state=seed+sim_num
    )
    smearing_reg.fit(X, y)
    
    y_pred_smearing = smearing_reg.predict(X)
    result.update({
        'insample_smearing': mean_squared_error(y_true, y_pred_smearing),
        'mse_smearing': mean_squared_error(y, y_pred_smearing),
        'coef_recovery_smearing': np.mean((
            np.mean([est.coef_ for est in smearing_reg.estimators_], axis=0) - beta_true
        )**2),
        'noise_sigma_smearing': best_sigma
    })
    
    ## Create penalized scorer factory with true sigma^2
    make_k_scorer = create_penalized_scorer(
        sigma=sigma,
        n=params['data']['n_train'],
        p=params['data']['n_predictors']
    )
    
    # Calculate m_grid
    m_grid = get_m_grid(
        params['model']['m_grid'],
        params['data']['n_predictors']
    )
    
    # Fit RGSCV with scorer factory
    rgscv = RGSCV(
        k_max=params['model']['k_max'],
        m_grid=m_grid,
        n_estimators=params['model']['rgscv']['n_estimators'],
        n_resample_iter=params['model']['rgscv']['n_resample_iter'],
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer
    )
    rgscv.fit(X, y)
    
    # Get predictions using best parameters
    y_pred_rgs = rgscv.predict(X)
    result.update({
        'best_m': rgscv.m_,
        'best_k': rgscv.k_,
        'insample_rgs': mean_squared_error(y_true, y_pred_rgs),
        'mse_rgs': mean_squared_error(y, y_pred_rgs),
        'coef_recovery_rgs': np.mean((rgscv.model_.coef_[rgscv.k_] - beta_true)**2),
        'support_recovery_rgs': np.mean(
            (np.abs(rgscv.model_.coef_[rgscv.k_]) > 1e-10) == (beta_true != 0)
        )
    })

    # Fit Greedy Selection
    gscv = RGSCV(
        k_max=params['model']['k_max'],
        m_grid=list([params['data']['n_predictors']]),
        n_estimators=1,
        n_resample_iter=0,
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer
    )
    gscv.fit(X, y)
    
    # Get predictions using best parameters
    y_pred_gs = gscv.predict(X)
    result.update({
        'best_k_gs': gscv.k_,  # Distinguish from RGS k
        'insample_gs': mean_squared_error(y_true, y_pred_gs),
        'mse_gs': mean_squared_error(y, y_pred_gs),
        'coef_recovery_gs': np.mean((gscv.model_.coef_[gscv.k_] - beta_true)**2),
        'support_recovery_gs': np.mean(
            (np.abs(gscv.model_.coef_[gscv.k_]) > 1e-10) == (beta_true != 0)
        )
    })
    
    return result

def main(param_path):
    """Main simulation loop."""
    # Load parameters
    params = load_params(param_path)
    start_time = time.time()
    
    # Generate base design matrix
    X_generators = {
        'orthogonal': generate_orthogonal_X,
        'banded': lambda n_predictors, n_train, seed: generate_banded_X(
            n_predictors=n_predictors,
            n_train=n_train,
            gamma=params['data']['banded_params']['gamma'],
            seed=seed
        ),
        'block': lambda n_predictors, n_train, seed: generate_block_X(
            n_predictors=n_predictors,
            n_train=n_train,
            block_size=params['data']['block_params']['block_size'],
            within_correlation=params['data']['block_params']['within_correlation'],
            seed=seed
        )
    }
    
    X = X_generators[params['data']['covariance_type']](
        params['data']['n_predictors'],
        params['data']['n_train'],
        params['simulation']['base_seed']
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
    sigmas = sorted(sigmas)  # Sort for nice progression in progress bar
    
    # Save the actual sigma values used
    params['simulation']['sigma']['computed_values'] = sigmas
    
    # Initialize results storage
    all_results = []
    
    # Main simulation loop with progress bar
    total_sims = params['simulation']['n_sim'] * len(sigmas)
    with tqdm(total=total_sims, desc="Total Progress") as pbar:
        for sim in range(1, params['simulation']['n_sim']+1):
            seed = params['simulation']['base_seed']
            
            for sigma in sigmas:
                pbar.set_description(
                    f"Sim {sim}/{params['simulation']['n_sim']}, "
                    f"σ={sigma}, {params['data']['generator_type']}, "
                    f"{params['data']['covariance_type']}"
                )
                
                result = run_one_dgp_iter(
                    X=X,
                    generator=generator,
                    sigma=sigma,
                    params=params,
                    seed=seed,
                    sim_num=sim
                )
                all_results.append(result)
                pbar.update(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    summary_metrics = {
        'best_m': ['mean', 'std'],
        'best_k': ['mean', 'std'],
        'best_k_gs': ['mean', 'std'],
        'insample_lasso': ['mean', 'std'],
        'mse_lasso': ['mean', 'std'],
        'coef_recovery_lasso': ['mean', 'std'],
        'support_recovery_lasso': ['mean', 'std'],
        'insample_ridge': ['mean', 'std'],
        'mse_ridge': ['mean', 'std'],
        'coef_recovery_ridge': ['mean', 'std'],
        'insample_elastic': ['mean', 'std'],
        'mse_elastic': ['mean', 'std'],
        'coef_recovery_elastic': ['mean', 'std'],
        'support_recovery_elastic': ['mean', 'std'],
        'insample_bagged': ['mean', 'std'],
        'mse_bagged': ['mean', 'std'],
        'coef_recovery_bagged': ['mean', 'std'],
        'max_samples_bagged': ['mean', 'std'],
        'max_features_bagged': ['mean', 'std'],
        'insample_smearing': ['mean', 'std'],
        'mse_smearing': ['mean', 'std'],
        'coef_recovery_smearing': ['mean', 'std'],
        'noise_sigma_smearing': ['mean', 'std'],
        'insample_rgs': ['mean', 'std'],
        'mse_rgs': ['mean', 'std'],
        'coef_recovery_rgs': ['mean', 'std'],
        'support_recovery_rgs': ['mean', 'std'],
        'insample_gs': ['mean', 'std'],
        'mse_gs': ['mean', 'std'],
        'coef_recovery_gs': ['mean', 'std'],
        'support_recovery_gs': ['mean', 'std']
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

    # Get the project root directory
    root_dir = Path(__file__).parent.parent  # Go up one level from scripts folder
    param_path = root_dir / "params" / "sim_params.json"
    results_df, summary = main(param_path)