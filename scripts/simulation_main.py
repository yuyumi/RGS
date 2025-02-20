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
from sklearn.model_selection import GridSearchCV, train_test_split

# Import from core RGS package
from rgs.core.rgs import RGS, RGSCV
from rgs.bogdan_penalty import create_bogdan_scorer
from rgs.aic_penalty import create_aic_scorer

# Import from simulation package
from rgs_experiments.utils.sim_util_dgs import *
from rgs_experiments.models.data_smearing import SmearedGS
from rgs_experiments.models.bagging import BaggedGS

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
    
    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X, 
        y,
        y_true,
        test_size=params['data']['test_size'],
        random_state=seed+sim_num,  # Set seed for reproducibility
    )

    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"y_true shape: {y_true.shape}")
    # print(f"X shape: {X_train.shape}")
    # print(f"y shape: {y_train.shape}")
    # print(f"y_true shape: {y_true_train.shape}")

    # Fit baseline models with specified CV
    cv = params['model']['baseline']['cv']
    
    # Initialize result dictionary
    result = {
        'simulation': sim_num,
        'sigma': sigma
    }
    
    # Fit and evaluate Lasso
    lasso = LassoCV(cv=cv, random_state=seed+sim_num)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_train)
    y_test_lasso = lasso.predict(X_test)
    result.update({
        'insample_lasso': mean_squared_error(y_true_train, y_pred_lasso),
        'mse_lasso': mean_squared_error(y_train, y_pred_lasso),
        'coef_recovery_lasso': np.mean((lasso.coef_ - beta_true)**2),
        'support_recovery_lasso': np.mean((lasso.coef_ != 0) == (beta_true != 0)),
        'outsample_mse_lasso': mean_squared_error(y_test, y_test_lasso)
    })
    
    # Fit and evaluate Ridge
    ridge = RidgeCV(cv=cv)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_train)
    y_test_ridge = ridge.predict(X_test)
    result.update({
        'insample_ridge': mean_squared_error(y_true_train, y_pred_ridge),
        'mse_ridge': mean_squared_error(y_train, y_pred_ridge),
        'coef_recovery_ridge': np.mean((ridge.coef_ - beta_true)**2),
        'outsample_mse_ridge': mean_squared_error(y_test, y_test_ridge)
    })
    
    # Fit and evaluate Elastic Net
    elastic = ElasticNetCV(cv=cv, random_state=seed+sim_num)
    elastic.fit(X_train, y_train)
    y_pred_elastic = elastic.predict(X_train)
    y_test_elastic = elastic.predict(X_test)
    result.update({
        'insample_elastic': mean_squared_error(y_true_train, y_pred_elastic),
        'mse_elastic': mean_squared_error(y_train, y_pred_elastic),
        'coef_recovery_elastic': np.mean((elastic.coef_ - beta_true)**2),
        'support_recovery_elastic': np.mean((elastic.coef_ != 0) == (beta_true != 0)),
        'outsample_mse_elastic': mean_squared_error(y_test, y_test_elastic)
    })

    ## Create penalized scorer factory with true sigma^2
    make_k_scorer = create_aic_scorer(
        sigma=sigma,
        n=params['data']['n_train'],
        p=params['data']['n_predictors']
    )

    # Fit and evaluate BaggedGS
    bagged_gs = BaggedGS(
        k_max=params['model']['k_max'],
        n_estimators=params['model']['bagged_gs']['n_estimators'],
        random_state=seed+sim_num,
        cv=params['model']['bagged_gs']['cv'],
        scoring=make_k_scorer
    )
    bagged_gs.fit(X_train, y_train)
    
    y_pred_bagged_gs = bagged_gs.predict(X_train)
    y_test_bagged_gs = bagged_gs.predict(X_test)
    
    # Get average coefficients - now correctly uses updated structure
    avg_coef_bagged_gs = np.zeros(X.shape[1])
    for coefs, _, _ in bagged_gs.estimators_:
        avg_coef_bagged_gs += coefs[bagged_gs.k_]
    avg_coef_bagged_gs /= len(bagged_gs.estimators_)
    
    result.update({
        'best_k_bagged_gs': bagged_gs.k_,
        'insample_bagged_gs': mean_squared_error(y_true_train, y_pred_bagged_gs),
        'mse_bagged_gs': mean_squared_error(y_train, y_pred_bagged_gs),
        'coef_recovery_bagged_gs': np.mean((avg_coef_bagged_gs - beta_true)**2),
        'support_recovery_bagged_gs': np.mean(
            (np.abs(avg_coef_bagged_gs) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_bagged_gs': mean_squared_error(y_test, y_test_bagged_gs)
    })
    
    # Fit and evaluate SmearedGS
    smeared_gs = SmearedGS(
        k_max=params['model']['k_max'],
        n_estimators=params['model']['smeared_gs']['n_estimators'],
        noise_scale=params['model']['smeared_gs']['param_grid']['noise_scale'],
        random_state=seed+sim_num,
        cv=params['model']['smeared_gs']['cv'],
        scoring=make_k_scorer
    )
    smeared_gs.fit(X_train, y_train)
    
    y_pred_smeared_gs = smeared_gs.predict(X_train)
    y_test_smeared_gs = smeared_gs.predict(X_test)
    
    # Get average coefficients - now correctly uses updated structure
    avg_coef_smeared_gs = np.zeros(X.shape[1])
    for coefs, _, _ in smeared_gs.estimators_:
        avg_coef_smeared_gs += coefs[smeared_gs.k_]
    avg_coef_smeared_gs /= len(smeared_gs.estimators_)
    
    result.update({
        'best_k_smeared_gs': smeared_gs.k_,
        'best_noise_scale': smeared_gs.noise_scale_,
        'insample_smeared_gs': mean_squared_error(y_true_train, y_pred_smeared_gs),
        'mse_smeared_gs': mean_squared_error(y_train, y_pred_smeared_gs),
        'coef_recovery_smeared_gs': np.mean((avg_coef_smeared_gs - beta_true)**2),
        'support_recovery_smeared_gs': np.mean(
            (np.abs(avg_coef_smeared_gs) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_smeared_gs': mean_squared_error(y_test, y_test_smeared_gs)
    })
    
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
    rgscv.fit(X_train, y_train)
    
    # Get predictions using best parameters
    y_pred_rgs = rgscv.predict(X_train)
    y_test_rgs = rgscv.predict(X_test)
    result.update({
        'best_m': rgscv.m_,
        'best_k': rgscv.k_,
        'insample_rgs': mean_squared_error(y_true_train, y_pred_rgs),
        'mse_rgs': mean_squared_error(y_train, y_pred_rgs),
        'coef_recovery_rgs': np.mean((rgscv.model_.coef_[rgscv.k_] - beta_true)**2),
        'support_recovery_rgs': np.mean(
            (np.abs(rgscv.model_.coef_[rgscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_rgs': mean_squared_error(y_test, y_test_rgs)
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
    gscv.fit(X_train, y_train)
    
    # Get predictions using best parameters
    y_pred_gs = gscv.predict(X_train)
    y_test_gs = gscv.predict(X_test)
    result.update({
        'best_k_original_gs': gscv.k_,  # Distinguish from RGS k
        'insample_original_gs': mean_squared_error(y_true_train, y_pred_gs),
        'mse_original_gs': mean_squared_error(y_train, y_pred_gs),
        'coef_recovery_original_gs': np.mean((gscv.model_.coef_[gscv.k_] - beta_true)**2),
        'support_recovery_original_gs': np.mean(
            (np.abs(gscv.model_.coef_[gscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_original_gs': mean_squared_error(y_test, y_test_gs)
    })

    # Baseline RGS and GS methods
    true_k = int(params['data']['signal_proportion']*params['data']['n_predictors'])
    base_rgscv = RGSCV(
        k_max=true_k,
        m_grid=m_grid,
        n_estimators=params['model']['rgscv']['n_estimators'],
        n_resample_iter=params['model']['rgscv']['n_resample_iter'],
        k_grid=[true_k],
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer
    )
    base_rgscv.fit(X_train, y_train)
    
    # Get predictions using best parameters
    y_pred_base_rgs = base_rgscv.predict(X_train)
    y_test_base_rgs = base_rgscv.predict(X_test)
    result.update({
        'best_m_base': base_rgscv.m_,
        'insample_base_rgs': mean_squared_error(y_true_train, y_pred_base_rgs),
        'mse_base_rgs': mean_squared_error(y_train, y_pred_base_rgs),
        'coef_recovery_base_rgs': np.mean((base_rgscv.model_.coef_[base_rgscv.k_] - beta_true)**2),
        'support_recovery_base_rgs': np.mean(
            (np.abs(base_rgscv.model_.coef_[base_rgscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_base_rgs': mean_squared_error(y_test, y_test_base_rgs)
    })

    # Fit Greedy Selection
    base_gscv = RGS(
        k_max=true_k,
        m=params['data']['n_predictors'],
        n_estimators=1,
        n_resample_iter=0,
        random_state=seed+sim_num
    )
    base_gscv.fit(X_train, y_train)
    
    # Get predictions using best parameters
    y_pred_base_gs = base_gscv.predict(X_train)
    y_test_base_gs = base_gscv.predict(X_test)
    result.update({
        'insample_base_gs': mean_squared_error(y_true_train, y_pred_base_gs),
        'mse_base_gs': mean_squared_error(y_train, y_pred_base_gs),
        'coef_recovery_base_gs': np.mean((base_gscv.coef_ - beta_true)**2),
        'support_recovery_base_gs': np.mean(
            (np.abs(base_gscv.coef_) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_base_gs': mean_squared_error(y_test, y_test_base_gs)
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
        'best_k_original_gs': ['mean', 'std'],
        'best_m_base': ['mean', 'std'],
        'insample_lasso': ['mean', 'std'],
        'mse_lasso': ['mean', 'std'],
        'coef_recovery_lasso': ['mean', 'std'],
        'support_recovery_lasso': ['mean', 'std'],
        'outsample_mse_lasso': ['mean', 'std'],  # Added
        'insample_ridge': ['mean', 'std'],
        'mse_ridge': ['mean', 'std'],
        'coef_recovery_ridge': ['mean', 'std'],
        'outsample_mse_ridge': ['mean', 'std'],  # Added
        'insample_elastic': ['mean', 'std'],
        'mse_elastic': ['mean', 'std'],
        'coef_recovery_elastic': ['mean', 'std'],
        'support_recovery_elastic': ['mean', 'std'],
        'outsample_mse_elastic': ['mean', 'std'],  # Added
        'best_k_bagged_gs': ['mean', 'std'],
        'insample_bagged_gs': ['mean', 'std'],
        'mse_bagged_gs': ['mean', 'std'],
        'coef_recovery_bagged_gs': ['mean', 'std'],
        'support_recovery_bagged_gs': ['mean', 'std'],
        'outsample_mse_bagged_gs': ['mean', 'std'],  # Added
        'best_k_smeared_gs': ['mean', 'std'],
        'best_noise_scale': ['mean', 'std'],
        'insample_smeared_gs': ['mean', 'std'],
        'mse_smeared_gs': ['mean', 'std'],
        'coef_recovery_smeared_gs': ['mean', 'std'],
        'support_recovery_smeared_gs': ['mean', 'std'],
        'outsample_mse_smeared_gs': ['mean', 'std'],  # Added
        'insample_rgs': ['mean', 'std'],
        'mse_rgs': ['mean', 'std'],
        'coef_recovery_rgs': ['mean', 'std'],
        'support_recovery_rgs': ['mean', 'std'],
        'outsample_mse_rgs': ['mean', 'std'],  # Added
        'insample_original_gs': ['mean', 'std'],
        'mse_original_gs': ['mean', 'std'],
        'coef_recovery_original_gs': ['mean', 'std'],
        'support_recovery_original_gs': ['mean', 'std'],
        'outsample_mse_original_gs': ['mean', 'std'],  # Added
        'insample_base_rgs': ['mean', 'std'],
        'mse_base_rgs': ['mean', 'std'],
        'coef_recovery_base_rgs': ['mean', 'std'],
        'support_recovery_base_rgs': ['mean', 'std'],
        'outsample_mse_base_rgs': ['mean', 'std'],  # Added
        'insample_base_gs': ['mean', 'std'],
        'mse_base_gs': ['mean', 'std'],
        'coef_recovery_base_gs': ['mean', 'std'],
        'support_recovery_base_gs': ['mean', 'std'],
        'outsample_mse_base_gs': ['mean', 'std']  # Added
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