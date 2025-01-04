import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from functools import partial
import warnings
import concurrent.futures
import multiprocessing
import random


def proj_span(M: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute projection using PyTorch."""
    if M.nelement() == 0:
        return torch.zeros_like(y)
    
    try:
        P = M @ torch.linalg.inv(M.T @ M) @ M.T
        return P @ y
    except RuntimeError:
        return torch.zeros_like(y)

def compute_penalty(k: int, m: int, p: int, omega: torch.Tensor, sigma2: float, n: int) -> float:
    """Compute penalty term using PyTorch."""
    return (2 * sigma2 / n) * sum(torch.log(torch.tensor(p / (j+1))) * omega[j] for j in range(k))

def perform_single_rfs(b: int, X_train: torch.Tensor, y_train: torch.Tensor, 
                      max_k: int, m: int, random_state: int = None) -> list:
    """Perform single RFS iteration using PyTorch."""
    n, p = X_train.shape
    torch.manual_seed(random_state + b)
    M_indices = []
    
    for i in range(max_k):
        available_features = list(set(range(p)) - set(M_indices))
        if len(available_features) < m:
            m = len(available_features)
        if m == 0:
            break
            
        perm = torch.randperm(len(available_features))
        A_i = torch.tensor([available_features[i] for i in perm[:m]])
        
        min_error = float('inf')
        best_idx = -1
        
        M_current = X_train[:, M_indices] if M_indices else torch.empty((n, 0), device=X_train.device)
        
        for j in A_i:
            M_candidate = torch.cat([M_current, X_train[:, j:j+1]], dim=1) if M_current.nelement() > 0 else X_train[:, j:j+1]
            proj = proj_span(M_candidate, y_train)
            error = torch.norm(y_train - proj).item()
            
            if error < min_error:
                min_error = error
                best_idx = j.item()
        
        if best_idx != -1:
            M_indices.append(best_idx)
    
    return M_indices

def optimized_rfs(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, 
                 max_k: int, B: int, optimize_m: bool = True, fit_intercept: bool = True, 
                 fs: bool = False) -> tuple:
    """RFS optimization using PyTorch."""
    n, p = X_train.shape
    best_penalized_error = float('inf')
    best_models = None
    sigma2 = torch.var(y_train).item()

    m_range = range(1, p + 1) if optimize_m else [p // 3]
    m_range = [p] if fs else m_range

    device = X_train.device

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for m in m_range:
            partial_rfs = partial(perform_single_rfs, X_train=X_train.cpu(), y_train=y_train.cpu(), 
                                max_k=max_k, m=m)
            
            futures = [executor.submit(partial_rfs, b, random_state=b) for b in range(B)]
            models = [future.result() for future in concurrent.futures.as_completed(futures)]

            omega = torch.tensor([sum(1 for model in models if j in model[:max_k]) / B for j in range(p)])
            
            for k in range(1, max_k + 1):
                f_b = torch.zeros_like(y_val)
                for model in models:
                    M_k = X_val[:, model[:k]]
                    f_b += proj_span(M_k, y_val)
                f_k_m_p_B = f_b / B

                penalty = compute_penalty(k, m, p, omega, sigma2, n)
                penalized_error = torch.norm(y_val - f_k_m_p_B).item() + penalty

                if penalized_error < best_penalized_error:
                    best_penalized_error = penalized_error
                    best_m = m
                    best_k = k
                    best_models = models

    return best_models, best_m, best_k

def apply_rfs(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, 
             models: list, k: int, fit_intercept: bool = True) -> torch.Tensor:
    """Apply RFS models using PyTorch."""
    predictions = torch.zeros(len(X_test), device=X_test.device)
    for model in models:
        X_train_subset = X_train[:, model[:k]]
        X_test_subset = X_test[:, model[:k]]

        if fit_intercept:
            ones_train = torch.ones(X_train_subset.shape[0], 1, device=X_train_subset.device)
            ones_test = torch.ones(X_test_subset.shape[0], 1, device=X_test_subset.device)
            X_train_subset = torch.cat([ones_train, X_train_subset], dim=1)
            X_test_subset = torch.cat([ones_test, X_test_subset], dim=1)

        lr = torch.linalg.lstsq(X_train_subset, y_train).solution
        predictions += X_test_subset @ lr

    return predictions / len(models)

def run_single_replication(sim, X_std, y, n_features, device='cuda'):
    random.seed(123+sim)
    X_std_torch = torch.tensor(X_std, dtype=torch.float32, device=device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)
    
    # Split data for PyTorch and sklearn models
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_std, y, test_size=0.7, random_state=sim)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=sim)
    
    # Create PyTorch tensors for RFS
    X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Sklearn models
    # Lasso
    lasso = Lasso(fit_intercept=True)
    alphas = np.logspace(-4, 1, 50)
    scores = [cross_val_score(Lasso(alpha=alpha), X_train, y_train, cv=5).mean()
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    lasso.set_params(alpha=best_alpha)
    lasso.fit(X_train, y_train)
    mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))

    # Elastic Net
    elastic = ElasticNet(fit_intercept=True)
    l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
    scores = [cross_val_score(ElasticNet(alpha=best_alpha, l1_ratio=ratio),
                              X_train, y_train, cv=5).mean()
              for ratio in l1_ratios]
    best_ratio = l1_ratios[np.argmax(scores)]
    elastic.set_params(alpha=best_alpha, l1_ratio=best_ratio)
    elastic.fit(X_train, y_train)
    mse_elastic = mean_squared_error(y_test, elastic.predict(X_test))

    # Ridge
    ridge = Ridge(fit_intercept=True)
    scores = [cross_val_score(Ridge(alpha=alpha), X_train, y_train, cv=5).mean()
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    ridge.set_params(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))

    # OLS
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train, y_train)
    mse_ols = mean_squared_error(y_test, ols.predict(X_test))

    # PyTorch RFS models
    rfs_models_unopt, best_m_unopt, best_k_unopt = optimized_rfs(X_train_torch, y_train_torch, 
                                                                X_val_torch, y_val_torch,
                                                                n_features, B=500, optimize_m=False)
    rfs_pred_unopt = apply_rfs(X_train_torch, y_train_torch, X_test_torch, 
                              rfs_models_unopt, best_k_unopt)
    mse_rfs_unopt = mean_squared_error(y_test, rfs_pred_unopt.cpu())

    rfs_models_opt, best_m_opt, best_k_opt = optimized_rfs(X_train_torch, y_train_torch, 
                                                          X_val_torch, y_val_torch,
                                                          n_features, B=500, optimize_m=True)
    rfs_pred_opt = apply_rfs(X_train_torch, y_train_torch, X_test_torch, 
                            rfs_models_opt, best_k_opt)
    mse_rfs_opt = mean_squared_error(y_test, rfs_pred_opt.cpu())

    fs_models, _, best_k_fs = optimized_rfs(X_train_torch, y_train_torch, 
                                          X_val_torch, y_val_torch,
                                          n_features, B=1, fs=True)
    fs_pred = apply_rfs(X_train_torch, y_train_torch, X_test_torch, 
                       fs_models, best_k_fs)
    mse_fs = mean_squared_error(y_test, fs_pred.cpu())

    return (mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, 
            mse_fs, mse_ols, best_m_opt, best_k_unopt, best_k_opt, best_k_fs)

def run_regression_comparison(X, y, label, n_simulations=50, device='cuda'):
    warnings.filterwarnings('ignore')
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(
                partial(run_single_replication, X_std=X_std, y=y, n_features=n_features, device=device),
                range(n_simulations)
            ),
            total=n_simulations,
            desc=f"Running Dataset {label}"
        ))

    return tuple(np.array(x) for x in zip(*results))

# Main execution code
labels = ['Auto Pricing', 'Bodyfat']
data = {}

# Load data
data['Auto Pricing'] = pd.read_csv('207_autoPrice.tsv', sep='\t')
data['Satellite Image'] = pd.read_csv('294_satellite_image.tsv', sep='\t')
data['Political'] = pd.read_csv('201_pol.tsv', sep='\t')
data['Bodyfat'] = pd.read_csv('560_bodyfat.tsv', sep='\t')

Xs = {label: data[label].drop('target', axis=1).values for label in labels}
ys = {label: data[label]['target'].values for label in labels}

# Run experiments
results = {label: run_regression_comparison(Xs[label], ys[label], label, n_simulations=50) 
          for label in labels}

# Save results
methods = ['Lasso', 'Elastic Net', 'Ridge', 'RFS Unopt', 'RFS Opt', 'FS', 'OLS']
mse_data = [
    {'Dataset': label, 'Method': method, 'MSE': mse}
    for label, body in results.items()
    for method, mses in zip(methods, body[:7])
    for mse in mses
]

best_ms = [
    {'Dataset': label, 'Best m': m}
    for label, body in results.items()
    for m in body[7]
]

best_ks = [
    {'Dataset': label, 'Method': method, 'Best k': k}
    for label, body in results.items()
    for method, ks in zip(methods[3:6], body[8:])
    for k in ks
]

pd.DataFrame(mse_data).to_csv('real_data_mse.csv', index=False)
pd.DataFrame(best_ms).to_csv('real_data_best_ms.csv', index=False)
pd.DataFrame(best_ks).to_csv('real_data_best_ks.csv', index=False)