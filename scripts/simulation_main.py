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
    
def check_matrix_rank(X):
    """
    Check if a matrix X is full rank and provide diagnostic information.
    
    Parameters
    ----------
    X : ndarray
        The design matrix to check
        
    Returns
    -------
    dict
        Dictionary containing rank information and diagnostics
    """
    
    n, p = X.shape
    min_dim = min(n, p)
    
    # Calculate rank using SVD
    # SVD is more numerically stable than np.linalg.matrix_rank
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    
    # Get the rank (number of non-zero singular values, with tolerance)
    tol = s.max() * max(X.shape) * np.finfo(s.dtype).eps
    rank = np.sum(s > tol)
    
    # Calculate condition number
    condition_number = s[0] / s[-1] if s[-1] > tol else np.inf
    
    # Check if X'X is positive definite
    is_pos_def = True
    try:
        # Try Cholesky decomposition - will fail if not positive definite
        np.linalg.cholesky(X.T @ X)
    except np.linalg.LinAlgError:
        is_pos_def = False
    
    # Prepare results
    results = {
        'is_full_rank': bool(rank == min_dim),  # Convert from numpy.bool_ to Python bool
        'rank': int(rank),  # Convert from numpy.int64 to Python int
        'min_dimension': int(min_dim),
        'condition_number': float(condition_number),
        'smallest_singular_value': float(s[-1]),
        'largest_singular_value': float(s[0]),
        'XTX_is_positive_definite': bool(is_pos_def)
    }
    return results

def calculate_df_for_all_k(model, X_train, y_train, y_true_train, sigma, n_train):
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

def calculate_df_for_all_k_ensemble(model, X_train, y_train, y_true_train, sigma, n_train):
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

def calculate_mse_for_all_k(model, X_train, y_train):
    """
    Calculate MSE for each k value in a fitted RGS model.
    
    Parameters
    ----------
    model : RGS or RGSCV
        The fitted model with coefficients for each k
    X_train : ndarray
        Training features
    y_train : ndarray
        Observed training targets
        
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
    
    # Calculate MSE for each k value
    mse_by_k = {}
    
    for k in range(len(rgs_model.coef_)):
        # Get predictions for this k
        y_pred_k = rgs_model.predict(X_train, k=k)
        
        # Calculate MSE (observed vs. predicted)
        mse_k = np.mean((y_train - y_pred_k) ** 2)
        
        # Store in dictionary
        mse_by_k[k] = mse_k
    
    return mse_by_k

def calculate_insample_for_all_k(model, X_train, y_true_train):
    """
    Calculate in-sample error for each k value in a fitted RGS model.
    
    Parameters
    ----------
    model : RGS or RGSCV
        The fitted model with coefficients for each k
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
    
    # Calculate in-sample error for each k value
    insample_by_k = {}
    
    for k in range(len(rgs_model.coef_)):
        # Get predictions for this k
        y_pred_k = rgs_model.predict(X_train, k=k)
        
        # Calculate in-sample error (true signal vs. predicted)
        insample_k = np.mean((y_true_train - y_pred_k) ** 2)
        
        # Store in dictionary
        insample_by_k[k] = insample_k
    
    return insample_by_k

def calculate_mse_for_all_k_ensemble(model, X_train, y_train):
    """
    Calculate MSE for each k value in a fitted ensemble model.
    
    Parameters
    ----------
    model : BaggedGS or SmearedGS
        The fitted ensemble model
    X_train : ndarray
        Training features
    y_train : ndarray
        Observed training targets
        
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

def calculate_insample_for_all_k_ensemble(model, X_train, y_true_train):
    """
    Calculate in-sample error for each k value in a fitted ensemble model.
    
    Parameters
    ----------
    model : BaggedGS or SmearedGS
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

def calculate_relative_test_error(beta_hat, beta_true, X_test, sigma, cov_matrix=None):
    """
    Calculate the Relative Test Error (RTE).
    
    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients
    beta_true : ndarray
        True coefficients
    X_test : ndarray
        Test design matrix
    sigma : float
        Noise standard deviation
    cov_matrix : ndarray, optional
        The covariance matrix of X. If None, it will be estimated from X_test.
        
    Returns
    -------
    float
        The relative test error (RTE)
    """
    # Calculate the covariance matrix if not provided
    if cov_matrix is None:
        cov_matrix = np.cov(X_test, rowvar=False)
    
    # Calculate the covariance-weighted squared difference between true and estimated betas
    beta_diff = beta_hat - beta_true
    weighted_beta_diff = beta_diff.T @ cov_matrix @ beta_diff
    
    # The RTE formula: (beta_hat - beta_true)^T Σ (beta_hat - beta_true) + sigma^2) / sigma^2
    rte = (weighted_beta_diff + sigma**2) / (sigma**2)
    
    return rte

def calculate_f_score(beta_hat, beta_true, threshold=1e-10):
    """
    Calculate F-score for support recovery.
    
    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients
    beta_true : ndarray
        True coefficients
    threshold : float
        Threshold for considering a coefficient non-zero
        
    Returns
    -------
    float
        F-score value (harmonic mean of precision and recall)
    """
    # Identify non-zero coefficients (true support)
    true_support = (np.abs(beta_true) > threshold)
    
    # Identify coefficients identified as non-zero by the model
    predicted_support = (np.abs(beta_hat) > threshold)
    
    # Calculate true positives, false positives, false negatives
    true_positives = np.sum(predicted_support & true_support)
    false_positives = np.sum(predicted_support & ~true_support)
    false_negatives = np.sum(~predicted_support & true_support)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f_score

def run_one_dgp_iter(
    X,
    cov_matrix,
    generator,
    sigma,
    params,
    seed,
    sim_num
):
    """Run one iteration of the simulation for a specific DGP setting."""
    # Dictionary to store timing information
    timing_results = {
        'simulation': sim_num,
        'sigma': sigma
    }
    
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

    # Number of training samples for df calculation
    n_train = y_train.shape[0]
    
    # Initialize result dictionary
    result = {
        'simulation': sim_num,
        'sigma': sigma
    }
    
    # Fit baseline models with specified CV
    cv = params['model']['baseline']['cv']
    
    # Fit and evaluate Lasso
    start_time = time.time()
    lasso = LassoCV(cv=cv, random_state=seed+sim_num)
    lasso.fit(X_train, y_train)
    lasso_time = time.time() - start_time
    timing_results['time_lasso'] = lasso_time
    
    y_pred_lasso = lasso.predict(X_train)
    y_test_lasso = lasso.predict(X_test)
    
    mse_lasso = mean_squared_error(y_train, y_pred_lasso)
    insample_lasso = mean_squared_error(y_true_train, y_pred_lasso)
    # Calculate degrees of freedom using the formula: df = (n/2*sigma^2) * (insample_error - mse + sigma^2)
    error_diff_lasso = insample_lasso - mse_lasso + sigma**2
    df_lasso = (n_train / (2 * sigma**2)) * error_diff_lasso

    rte_lasso = calculate_relative_test_error(
        beta_hat=lasso.coef_,
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_lasso = calculate_f_score(
        beta_hat=lasso.coef_,
        beta_true=beta_true
    )
    
    result.update({
        'insample_lasso': insample_lasso,
        'mse_lasso': mse_lasso,
        'df_lasso': df_lasso,
        'coef_recovery_lasso': np.mean((lasso.coef_ - beta_true)**2),
        'support_recovery_lasso': np.mean((lasso.coef_ != 0) == (beta_true != 0)),
        'outsample_mse_lasso': mean_squared_error(y_test, y_test_lasso),
        'rte_lasso': rte_lasso,
        'f_score_lasso': f_score_lasso
    })
    
    # Fit and evaluate Ridge
    start_time = time.time()
    ridge = RidgeCV(cv=cv)
    ridge.fit(X_train, y_train)
    ridge_time = time.time() - start_time
    timing_results['time_ridge'] = ridge_time
    
    y_pred_ridge = ridge.predict(X_train)
    y_test_ridge = ridge.predict(X_test)
    
    mse_ridge = mean_squared_error(y_train, y_pred_ridge)
    insample_ridge = mean_squared_error(y_true_train, y_pred_ridge)
    error_diff_ridge = insample_ridge - mse_ridge + sigma**2
    df_ridge = (n_train / (2 * sigma**2)) * error_diff_ridge

    rte_ridge = calculate_relative_test_error(
        beta_hat=ridge.coef_,
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_ridge = calculate_f_score(
        beta_hat=ridge.coef_,
        beta_true=beta_true
    )
    
    result.update({
        'insample_ridge': insample_ridge,
        'mse_ridge': mse_ridge,
        'df_ridge': df_ridge,
        'coef_recovery_ridge': np.mean((ridge.coef_ - beta_true)**2),
        'outsample_mse_ridge': mean_squared_error(y_test, y_test_ridge),
        'rte_ridge': rte_ridge,
        'f_score_ridge': f_score_ridge
    })
    
    # Fit and evaluate Elastic Net
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    start_time = time.time()
    elastic = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, random_state=seed+sim_num)
    elastic.fit(X_train, y_train)
    elastic_time = time.time() - start_time
    timing_results['time_elastic'] = elastic_time
    
    y_pred_elastic = elastic.predict(X_train)
    y_test_elastic = elastic.predict(X_test)
    
    mse_elastic = mean_squared_error(y_train, y_pred_elastic)
    insample_elastic = mean_squared_error(y_true_train, y_pred_elastic)
    error_diff_elastic = insample_elastic - mse_elastic + sigma**2
    df_elastic = (n_train / (2 * sigma**2)) * error_diff_elastic

    rte_elastic = calculate_relative_test_error(
        beta_hat=elastic.coef_,
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_elastic = calculate_f_score(
        beta_hat=elastic.coef_,
        beta_true=beta_true
    )
    
    result.update({
        'insample_elastic': insample_elastic,
        'mse_elastic': mse_elastic,
        'df_elastic': df_elastic,
        'coef_recovery_elastic': np.mean((elastic.coef_ - beta_true)**2),
        'support_recovery_elastic': np.mean((elastic.coef_ != 0) == (beta_true != 0)),
        'outsample_mse_elastic': mean_squared_error(y_test, y_test_elastic),
        'rte_elastic': rte_elastic,
        'f_score_elastic': f_score_elastic
    })

    # Create penalized scorer factory with true sigma^2
    make_k_scorer = create_aic_scorer(
        sigma=sigma,
        n=params['data']['n_train'],
        p=params['data']['n_predictors']
    )

    # Fit and evaluate BaggedGS
    start_time = time.time()
    bagged_gs = BaggedGS(
        k_max=params['model']['k_max'],
        n_estimators=params['model']['bagged_gs']['n_estimators'],
        random_state=seed+sim_num,
        cv=params['model']['bagged_gs']['cv'],
        scoring=make_k_scorer  # Use AIC scorer
    )
    bagged_gs.fit(X_train, y_train)
    bagged_gs_time = time.time() - start_time
    timing_results['time_bagged_gs'] = bagged_gs_time
    
    y_pred_bagged_gs = bagged_gs.predict(X_train)
    y_test_bagged_gs = bagged_gs.predict(X_test)
    
    # Get average coefficients
    avg_coef_bagged_gs = np.zeros(X.shape[1])
    for coefs, _, _ in bagged_gs.estimators_:
        avg_coef_bagged_gs += coefs[bagged_gs.k_]
    avg_coef_bagged_gs /= len(bagged_gs.estimators_)
    
    mse_bagged_gs = mean_squared_error(y_train, y_pred_bagged_gs)
    insample_bagged_gs = mean_squared_error(y_true_train, y_pred_bagged_gs)
    error_diff_bagged_gs = insample_bagged_gs - mse_bagged_gs + sigma**2
    df_bagged_gs = (n_train / (2 * sigma**2)) * error_diff_bagged_gs

    rte_bagged_gs = calculate_relative_test_error(
        beta_hat=avg_coef_bagged_gs,
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_bagged_gs = calculate_f_score(
        beta_hat=avg_coef_bagged_gs,
        beta_true=beta_true
    )
    
    result.update({
        'best_k_bagged_gs': bagged_gs.k_,
        'insample_bagged_gs': insample_bagged_gs,
        'mse_bagged_gs': mse_bagged_gs,
        'df_bagged_gs': df_bagged_gs,
        'coef_recovery_bagged_gs': np.mean((avg_coef_bagged_gs - beta_true)**2),
        'support_recovery_bagged_gs': np.mean(
            (np.abs(avg_coef_bagged_gs) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_bagged_gs': mean_squared_error(y_test, y_test_bagged_gs),
        'rte_bagged_gs': rte_bagged_gs,
        'f_score_bagged_gs': f_score_bagged_gs
    })

    df_by_k_bagged_gs = calculate_df_for_all_k_ensemble(
        model=bagged_gs,
        X_train=X_train, 
        y_train=y_train,
        y_true_train=y_true_train,
        sigma=sigma,
        n_train=n_train
    )

    mse_by_k_bagged_gs = calculate_mse_for_all_k_ensemble(
        model=bagged_gs,
        X_train=X_train, 
        y_train=y_train
    )

    insample_by_k_bagged_gs = calculate_insample_for_all_k_ensemble(
        model=bagged_gs,
        X_train=X_train, 
        y_true_train=y_true_train
    )

    for k, mse_value in mse_by_k_bagged_gs.items():
        result[f'mse_by_k_bagged_gs_{k}'] = mse_value
        
    for k, insample_value in insample_by_k_bagged_gs.items():
        result[f'insample_by_k_bagged_gs_{k}'] = insample_value

    for k, df_value in df_by_k_bagged_gs.items():
        result[f'df_by_k_bagged_gs_{k}'] = df_value
    
    # Fit and evaluate SmearedGS
    start_time = time.time()
    smeared_gs = SmearedGS(
        k_max=params['model']['k_max'],
        n_estimators=params['model']['smeared_gs']['n_estimators'],
        noise_scale=params['model']['smeared_gs']['param_grid']['noise_scale'],
        random_state=seed+sim_num,
        cv=params['model']['smeared_gs']['cv'],
        scoring=make_k_scorer  # Use AIC scorer
    )
    smeared_gs.fit(X_train, y_train)
    smeared_gs_time = time.time() - start_time
    timing_results['time_smeared_gs'] = smeared_gs_time
    
    y_pred_smeared_gs = smeared_gs.predict(X_train)
    y_test_smeared_gs = smeared_gs.predict(X_test)
    
    # Get average coefficients
    avg_coef_smeared_gs = np.zeros(X.shape[1])
    for coefs, _, _ in smeared_gs.estimators_:
        avg_coef_smeared_gs += coefs[smeared_gs.k_]
    avg_coef_smeared_gs /= len(smeared_gs.estimators_)
    
    mse_smeared_gs = mean_squared_error(y_train, y_pred_smeared_gs)
    insample_smeared_gs = mean_squared_error(y_true_train, y_pred_smeared_gs)
    error_diff_smeared_gs = insample_smeared_gs - mse_smeared_gs + sigma**2
    df_smeared_gs = (n_train / (2 * sigma**2)) * error_diff_smeared_gs

    rte_smeared_gs = calculate_relative_test_error(
        beta_hat=avg_coef_smeared_gs,
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_smeared_gs = calculate_f_score(
        beta_hat=avg_coef_smeared_gs,
        beta_true=beta_true
    )
    
    result.update({
        'best_k_smeared_gs': smeared_gs.k_,
        'best_noise_scale': smeared_gs.noise_scale_,
        'insample_smeared_gs': insample_smeared_gs,
        'mse_smeared_gs': mse_smeared_gs,
        'df_smeared_gs': df_smeared_gs,
        'coef_recovery_smeared_gs': np.mean((avg_coef_smeared_gs - beta_true)**2),
        'support_recovery_smeared_gs': np.mean(
            (np.abs(avg_coef_smeared_gs) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_smeared_gs': mean_squared_error(y_test, y_test_smeared_gs),
        'rte_smeared_gs': rte_smeared_gs,
        'f_score_smeared_gs': f_score_smeared_gs
    })

    df_by_k_smeared_gs = calculate_df_for_all_k_ensemble(
        model=smeared_gs,
        X_train=X_train, 
        y_train=y_train,
        y_true_train=y_true_train,
        sigma=sigma,
        n_train=n_train
    )

    mse_by_k_smeared_gs = calculate_mse_for_all_k_ensemble(
        model=smeared_gs,
        X_train=X_train, 
        y_train=y_train
    )

    insample_by_k_smeared_gs = calculate_insample_for_all_k_ensemble(
        model=smeared_gs,
        X_train=X_train, 
        y_true_train=y_true_train
    )

    for k, mse_value in mse_by_k_smeared_gs.items():
        result[f'mse_by_k_smeared_gs_{k}'] = mse_value
        
    for k, insample_value in insample_by_k_smeared_gs.items():
        result[f'insample_by_k_smeared_gs_{k}'] = insample_value

    for k, df_value in df_by_k_smeared_gs.items():
        result[f'df_by_k_smeared_gs_{k}'] = df_value
    
    # Calculate m_grid
    m_grid = get_m_grid(
        params['model']['m_grid'],
        params['data']['n_predictors']
    )

    # Fit RGSCV
    start_time = time.time()
    rgscv = RGSCV(
        k_max=params['model']['k_max'],
        m_grid=m_grid,
        n_estimators=params['model']['rgscv']['n_estimators'],
        n_resample_iter=params['model']['rgscv']['n_resample_iter'],
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer  # Use AIC scorer
    )
    rgscv.fit(X_train, y_train)
    rgscv_time = time.time() - start_time
    timing_results['time_rgscv'] = rgscv_time
    
    # Get predictions using best parameters
    y_pred_rgs = rgscv.predict(X_train)
    y_test_rgs = rgscv.predict(X_test)
    
    mse_rgs = mean_squared_error(y_train, y_pred_rgs)
    insample_rgs = mean_squared_error(y_true_train, y_pred_rgs)
    error_diff_rgs = insample_rgs - mse_rgs + sigma**2
    df_rgs = (n_train / (2 * sigma**2)) * error_diff_rgs

    rte_rgs = calculate_relative_test_error(
        beta_hat=rgscv.model_.coef_[rgscv.k_],
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_rgs = calculate_f_score(
        beta_hat=rgscv.model_.coef_[rgscv.k_],
        beta_true=beta_true
    )
    
    result.update({
        'best_m': rgscv.m_,
        'best_k': rgscv.k_,
        'insample_rgs': insample_rgs,
        'mse_rgs': mse_rgs,
        'df_rgs': df_rgs,
        'coef_recovery_rgs': np.mean((rgscv.model_.coef_[rgscv.k_] - beta_true)**2),
        'support_recovery_rgs': np.mean(
            (np.abs(rgscv.model_.coef_[rgscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_rgs': mean_squared_error(y_test, y_test_rgs),
        'rte_rgs': rte_rgs,
        'f_score_rgs': f_score_rgs
    })

    df_by_k_rgs = calculate_df_for_all_k(
        model=rgscv,
        X_train=X_train, 
        y_train=y_train,
        y_true_train=y_true_train,
        sigma=sigma,
        n_train=n_train
    )

    mse_by_k_rgs = calculate_mse_for_all_k(
        model=rgscv,
        X_train=X_train, 
        y_train=y_train
    )

    insample_by_k_rgs = calculate_insample_for_all_k(
        model=rgscv,
        X_train=X_train, 
        y_true_train=y_true_train
    )

    for k, mse_value in mse_by_k_rgs.items():
        result[f'mse_by_k_rgs_{k}'] = mse_value
        
    for k, insample_value in insample_by_k_rgs.items():
        result[f'insample_by_k_rgs_{k}'] = insample_value

    # Store in result dictionary with a consistent column naming pattern
    for k, df_value in df_by_k_rgs.items():
        result[f'df_by_k_rgs_{k}'] = df_value

    # Fit Greedy Selection
    start_time = time.time()
    gscv = RGSCV(
        k_max=params['model']['k_max'],
        m_grid=list([params['data']['n_predictors']]),
        n_estimators=1,
        n_resample_iter=0,
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer  # Use AIC scorer
    )
    gscv.fit(X_train, y_train)
    original_gs_time = time.time() - start_time
    timing_results['time_original_gs'] = original_gs_time
    
    # Get predictions using best parameters
    y_pred_gs = gscv.predict(X_train)
    y_test_gs = gscv.predict(X_test)
    
    mse_original_gs = mean_squared_error(y_train, y_pred_gs)
    insample_original_gs = mean_squared_error(y_true_train, y_pred_gs)
    error_diff_original_gs = insample_original_gs - mse_original_gs + sigma**2
    df_original_gs = (n_train / (2 * sigma**2)) * error_diff_original_gs

    rte_gs = calculate_relative_test_error(
        beta_hat=gscv.model_.coef_[gscv.k_],
        beta_true=beta_true,
        X_test=X_test,
        sigma=sigma,
        cov_matrix=cov_matrix
    )

    f_score_gs = calculate_f_score(
        beta_hat=gscv.model_.coef_[gscv.k_],
        beta_true=beta_true
    )
    
    result.update({
        'best_k_original_gs': gscv.k_,
        'insample_original_gs': insample_original_gs,
        'mse_original_gs': mse_original_gs,
        'df_original_gs': df_original_gs,
        'coef_recovery_original_gs': np.mean((gscv.model_.coef_[gscv.k_] - beta_true)**2),
        'support_recovery_original_gs': np.mean(
            (np.abs(gscv.model_.coef_[gscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_original_gs': mean_squared_error(y_test, y_test_gs),
        'rte_original_gs': rte_gs,
        'f_score_original_gs': f_score_gs
    })

    df_by_k_original_gs = calculate_df_for_all_k(
        model=gscv,
        X_train=X_train, 
        y_train=y_train,
        y_true_train=y_true_train,
        sigma=sigma,
        n_train=n_train
    )

    mse_by_k_original_gs = calculate_mse_for_all_k(
        model=gscv,
        X_train=X_train, 
        y_train=y_train
    )

    insample_by_k_original_gs = calculate_insample_for_all_k(
        model=gscv,
        X_train=X_train, 
        y_true_train=y_true_train
    )

    for k, mse_value in mse_by_k_original_gs.items():
        result[f'mse_by_k_original_gs_{k}'] = mse_value
        
    for k, insample_value in insample_by_k_original_gs.items():
        result[f'insample_by_k_original_gs_{k}'] = insample_value

    for k, df_value in df_by_k_original_gs.items():
        result[f'df_by_k_original_gs_{k}'] = df_value

    # Baseline RGS and GS methods
    true_k = int(params['data']['signal_proportion']*params['data']['n_predictors'])
    start_time = time.time()
    base_rgscv = RGSCV(
        k_max=true_k,
        m_grid=m_grid,
        n_estimators=params['model']['rgscv']['n_estimators'],
        n_resample_iter=params['model']['rgscv']['n_resample_iter'],
        k_grid=[true_k],
        random_state=seed+sim_num,
        cv=params['model']['rgscv']['cv'],
        scoring=make_k_scorer  # Use AIC scorer
    )
    base_rgscv.fit(X_train, y_train)
    base_rgs_time = time.time() - start_time
    timing_results['time_base_rgs'] = base_rgs_time
    
    # Get predictions using best parameters
    y_pred_base_rgs = base_rgscv.predict(X_train)
    y_test_base_rgs = base_rgscv.predict(X_test)
    
    mse_base_rgs = mean_squared_error(y_train, y_pred_base_rgs)
    insample_base_rgs = mean_squared_error(y_true_train, y_pred_base_rgs)
    error_diff_base_rgs = insample_base_rgs - mse_base_rgs + sigma**2
    df_base_rgs = (n_train / (2 * sigma**2)) * error_diff_base_rgs
    
    result.update({
        'best_m_base': base_rgscv.m_,
        'insample_base_rgs': insample_base_rgs,
        'mse_base_rgs': mse_base_rgs,
        'df_base_rgs': df_base_rgs,
        'coef_recovery_base_rgs': np.mean((base_rgscv.model_.coef_[base_rgscv.k_] - beta_true)**2),
        'support_recovery_base_rgs': np.mean(
            (np.abs(base_rgscv.model_.coef_[base_rgscv.k_]) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_base_rgs': mean_squared_error(y_test, y_test_base_rgs)
    })

    # Fit Greedy Selection
    start_time = time.time()
    base_gscv = RGS(
        k_max=true_k,
        m=params['data']['n_predictors'],
        n_estimators=1,
        n_resample_iter=0,
        random_state=seed+sim_num
    )
    base_gscv.fit(X_train, y_train)
    base_gs_time = time.time() - start_time
    timing_results['time_base_gs'] = base_gs_time
    
    # Get predictions using best parameters
    y_pred_base_gs = base_gscv.predict(X_train)
    y_test_base_gs = base_gscv.predict(X_test)
    
    mse_base_gs = mean_squared_error(y_train, y_pred_base_gs)
    insample_base_gs = mean_squared_error(y_true_train, y_pred_base_gs)
    error_diff_base_gs = insample_base_gs - mse_base_gs + sigma**2
    df_base_gs = (n_train / (2 * sigma**2)) * error_diff_base_gs
    
    result.update({
        'insample_base_gs': insample_base_gs,
        'mse_base_gs': mse_base_gs,
        'df_base_gs': df_base_gs,
        'coef_recovery_base_gs': np.mean((base_gscv.coef_ - beta_true)**2),
        'support_recovery_base_gs': np.mean(
            (np.abs(base_gscv.coef_) > 1e-10) == (beta_true != 0)
        ),
        'outsample_mse_base_gs': mean_squared_error(y_test, y_test_base_gs)
    })
    
    return result, timing_results
    

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
    
    X, cov_matrix = X_generators[params['data']['covariance_type']](
        params['data']['n_predictors'],
        params['data']['n_train'],
        params['simulation']['base_seed']
    )
    
    # Check if X is full rank and save the information
    rank_info = check_matrix_rank(X)
    
    # Log the rank information
    print(f"Rank check for design matrix X:")
    print(f"  - Dimensions: {X.shape}")
    print(f"  - Full rank: {rank_info['is_full_rank']}")
    print(f"  - Rank: {rank_info['rank']} / {rank_info['min_dimension']}")
    print(f"  - Condition number: {rank_info['condition_number']:.4e}")
    
    # If not full rank, provide more detailed information
    if not rank_info['is_full_rank']:
        print("WARNING: X is not full rank!")
        print(f"  - Smallest singular value: {rank_info['smallest_singular_value']:.4e}")
        print(f"  - X'X is positive definite: {rank_info['XTX_is_positive_definite']}")
        print("  - This may lead to unstable or non-unique solutions.")
    
    # Add rank information to parameters to save with results
    params['rank_check'] = rank_info
    
    # Set up generator mapping
    generators = {
        'exact': generate_exact_sparsity_example,
        'spaced': generate_spaced_sparsity_example,
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
    all_timing_results = []
    
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
                
                result, timing_result = run_one_dgp_iter(
                    X=X,
                    cov_matrix=cov_matrix,
                    generator=generator,
                    sigma=sigma,
                    params=params,
                    seed=seed,
                    sim_num=sim
                )
                all_results.append(result)
                all_timing_results.append(timing_result)
                pbar.update(1)
    
    # Convert results to DataFrames
    results_df = pd.DataFrame(all_results)
    timing_df = pd.DataFrame(all_timing_results)
    
    # Calculate summary statistics for timing
    timing_summary_metrics = {
        'time_lasso': ['mean', 'std', 'min', 'max'],
        'time_ridge': ['mean', 'std', 'min', 'max'],
        'time_elastic': ['mean', 'std', 'min', 'max'],
        'time_bagged_gs': ['mean', 'std', 'min', 'max'],
        'time_smeared_gs': ['mean', 'std', 'min', 'max'],
        'time_rgscv': ['mean', 'std', 'min', 'max'],
        'time_original_gs': ['mean', 'std', 'min', 'max'],
        'time_base_rgs': ['mean', 'std', 'min', 'max'],
        'time_base_gs': ['mean', 'std', 'min', 'max']
    }
    
    timing_summary = timing_df.groupby('sigma').agg(timing_summary_metrics).round(4)
    
    # Calculate performance summary statistics
    summary_metrics = {
        'best_m': ['mean', 'std'],
        'best_k': ['mean', 'std'],
        'best_k_original_gs': ['mean', 'std'],
        'best_m_base': ['mean', 'std'],
        'insample_lasso': ['mean', 'std'],
        'mse_lasso': ['mean', 'std'],
        'df_lasso': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_lasso': ['mean', 'std'],
        'support_recovery_lasso': ['mean', 'std'],
        'outsample_mse_lasso': ['mean', 'std'],
        'rte_lasso': ['mean', 'std'],
        'f_score_lasso': ['mean', 'std'],
        'insample_ridge': ['mean', 'std'],
        'mse_ridge': ['mean', 'std'],
        'df_ridge': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_ridge': ['mean', 'std'],
        'outsample_mse_ridge': ['mean', 'std'],
        'rte_ridge': ['mean', 'std'],
        'f_score_ridge': ['mean', 'std'],
        'insample_elastic': ['mean', 'std'],
        'mse_elastic': ['mean', 'std'],
        'df_elastic': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_elastic': ['mean', 'std'],
        'support_recovery_elastic': ['mean', 'std'],
        'outsample_mse_elastic': ['mean', 'std'],
        'rte_elastic': ['mean', 'std'],
        'f_score_elastic': ['mean', 'std'],
        'best_k_bagged_gs': ['mean', 'std'],
        'insample_bagged_gs': ['mean', 'std'],
        'mse_bagged_gs': ['mean', 'std'],
        'df_bagged_gs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_bagged_gs': ['mean', 'std'],
        'support_recovery_bagged_gs': ['mean', 'std'],
        'outsample_mse_bagged_gs': ['mean', 'std'],
        'rte_bagged_gs': ['mean', 'std'],
        'f_score_bagged_gs': ['mean', 'std'],
        'best_k_smeared_gs': ['mean', 'std'],
        'best_noise_scale': ['mean', 'std'],
        'insample_smeared_gs': ['mean', 'std'],
        'mse_smeared_gs': ['mean', 'std'],
        'df_smeared_gs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_smeared_gs': ['mean', 'std'],
        'support_recovery_smeared_gs': ['mean', 'std'],
        'outsample_mse_smeared_gs': ['mean', 'std'],
        'rte_smeared_gs': ['mean', 'std'],
        'f_score_smeared_gs': ['mean', 'std'],
        'insample_rgs': ['mean', 'std'],
        'mse_rgs': ['mean', 'std'],
        'df_rgs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_rgs': ['mean', 'std'],
        'support_recovery_rgs': ['mean', 'std'],
        'outsample_mse_rgs': ['mean', 'std'],
        'rte_rgs': ['mean', 'std'],
        'f_score_rgs': ['mean', 'std'],
        'insample_original_gs': ['mean', 'std'],
        'mse_original_gs': ['mean', 'std'],
        'df_original_gs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_original_gs': ['mean', 'std'],
        'support_recovery_original_gs': ['mean', 'std'],
        'outsample_mse_original_gs': ['mean', 'std'],
        'rte_original_gs': ['mean', 'std'],
        'f_score_original_gs': ['mean', 'std'],
        'insample_base_rgs': ['mean', 'std'],
        'mse_base_rgs': ['mean', 'std'],
        'df_base_rgs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_base_rgs': ['mean', 'std'],
        'support_recovery_base_rgs': ['mean', 'std'],
        'outsample_mse_base_rgs': ['mean', 'std'],
        'insample_base_gs': ['mean', 'std'],
        'mse_base_gs': ['mean', 'std'],
        'df_base_gs': ['mean', 'std'],  # Degrees of freedom
        'coef_recovery_base_gs': ['mean', 'std'],
        'support_recovery_base_gs': ['mean', 'std'],
        'outsample_mse_base_gs': ['mean', 'std']
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

    # Save results, timing, and summaries
    results_df.to_csv(save_path / f'simulation_results_{filename_base}.csv', index=False)
    timing_df.to_csv(save_path / f'simulation_timing_{filename_base}.csv', index=False)
    summary.to_csv(save_path / f'simulation_summary_{filename_base}.csv')
    timing_summary.to_csv(save_path / f'simulation_timing_summary_{filename_base}.csv')

    # Save parameters used
    with open(save_path / f'simulation_params_{filename_base}.json', 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\nSimulation completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved with base filename: {filename_base}")
    
    return results_df, summary, timing_df, timing_summary

if __name__ == "__main__":
    # Get the project root directory
    root_dir = Path(__file__).parent.parent  # Go up one level from scripts folder
    param_path = root_dir / "params" / "sim_params.json"
    results_df, summary, timing_df, timing_summary = main(param_path)