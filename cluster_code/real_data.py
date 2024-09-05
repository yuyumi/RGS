import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing
from functools import partial
import warnings
import torch


def optimized_rfs(X_train, y_train, X_val, y_val, max_k, B, optimize_m=True, fit_intercept=True, fs=False):
    """
    Perform Randomized Forward Selection with optional optimization of 'm' parameter on GPU.
    This version builds on the model incrementally for each k and handles both numpy arrays and PyTorch tensors.

    Parameters:
    X_train, y_train: Training data (numpy arrays or PyTorch tensors)
    X_val, y_val: Validation data (numpy arrays or PyTorch tensors)
    max_k: Maximum number of features to select
    B: Number of random subsets
    optimize_m: If True, optimize over all m; if False, set m = floor(p/3)

    Returns:
    best_models: List of selected feature indices for each random subset
    best_m: Optimal value for 'm' (or floor(p/3) if optimize_m is False)
    best_k: Optimal value for 'k'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to PyTorch tensors if inputs are numpy arrays
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    if isinstance(X_val, np.ndarray):
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    if isinstance(y_val, np.ndarray):
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

    n, p = X_train.shape
    best_mse = float('inf')
    best_models = None

    if optimize_m:
        m_range = range(1, p + 1)
    else:
        m_range = [p // 3]

    if fs:
        m_range = [p]

    for m in m_range:
        models = [[] for _ in range(B)]
        residuals = [y_train.clone() for _ in range(B)]
        coefficients = [torch.tensor([], device=device) for _ in range(B)]

        for k in range(1, max_k+1):
            for b in range(B):
                available_features = [i for i in range(p) if i not in models[b]]
                candidates = torch.tensor(np.random.choice(available_features, 
                                          size=min(m, p-len(models[b])), replace=False), 
                                          device=device)
                inner_products = torch.abs(torch.sum(X_train[:, candidates] * residuals[b].unsqueeze(1), dim=0))
                best_feature = candidates[torch.argmax(inner_products)]

                models[b].append(best_feature.item())
                X_new = X_train[:, best_feature].reshape(-1, 1)
                if k == 1:
                    coefficients[b] = torch.linalg.lstsq(X_new, y_train).solution
                else:
                    X_prev = X_train[:, models[b][:-1]]
                    X_combined = torch.cat((X_prev, X_new), dim=1)
                    coefficients[b] = torch.linalg.lstsq(X_combined, y_train).solution

                residuals[b] = y_train - X_train[:, models[b]] @ coefficients[b]

        # Evaluate the model on validation data
        mse_list = [mean_squared_error(y_val.cpu().numpy(), 
                    apply_rfs(X_train, y_train, X_val, models, k, fit_intercept).cpu().numpy())
                    for k in range(1, max_k + 1)]
        mse = min(mse_list)
        new_k = np.argmin(mse_list) + 1

        if mse < best_mse:
            best_mse = mse
            best_m = m
            best_k = new_k
            best_models = models

    return best_models, best_m, best_k

def apply_rfs(X_train, y_train, X_val, models, k, fit_intercept=True):
    """
    Apply the RFS model to validation data.
    """
    device = X_train.device
    selected_features = models[0][:k]  # Use the first model's features
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    
    if fit_intercept:
        X_train_selected = torch.cat([torch.ones(X_train_selected.shape[0], 1, device=device), X_train_selected], dim=1)
        X_val_selected = torch.cat([torch.ones(X_val_selected.shape[0], 1, device=device), X_val_selected], dim=1)
    
    coefficients = torch.linalg.lstsq(X_train_selected, y_train).solution
    return X_val_selected @ coefficients

# def optimized_rfs(X_train, y_train, X_val, y_val, max_k, B, optimize_m=True, fit_intercept=True, fs=False):
#     """
#     Perform Randomized Forward Selection with optional optimization of 'm' parameter.
#     This version builds on the model incrementally for each k.

#     Parameters:
#     X_train, y_train: Training data
#     X_val, y_val: Validation data
#     max_k: Maximum number of features to select
#     B: Number of random subsets
#     optimize_m: If True, optimize over all m; if False, set m = floor(p/3)

#     Returns:
#     best_models: List of selected feature indices for each random subset
#     best_m: Optimal value for 'm' (or floor(p/3) if optimize_m is False)
#     """
#     n, p = X_train.shape
#     best_mse = float('inf')
#     best_models = None

#     if optimize_m:
#         m_range = range(1, p + 1)
#     else:
#         m_range = [p // 3]

#     if fs:
#         m_range = [ p ]

#     for m in m_range:
#         models = [[] for _ in range(B)]
#         residuals = [y_train.copy() for _ in range(B)]
#         coefficients = [np.array([]) for _ in range(B)]

#         for k in range(1, max_k+1):
#             for b in range(B):
#                 candidates = np.random.choice([i for i in range(p) if i not in models[b]],
#                                               size=min(m, p-len(models[b])), replace=False)
#                 inner_products = np.abs([np.dot(X_train[:, j], residuals[b]) for j in candidates])
#                 best_feature = candidates[np.argmax(inner_products)]

#                 models[b].append(best_feature)
#                 X_new = X_train[:, best_feature].reshape(-1, 1)
#                 if k == 1:
#                     coefficients[b] = np.linalg.lstsq(X_new, y_train, rcond=None)[0]
#                 else:
#                     X_prev = X_train[:, models[b][:-1]]
#                     X_combined = np.column_stack((X_prev, X_new))
#                     coefficients[b] = np.linalg.lstsq(X_combined, y_train, rcond=None)[0]

#                 residuals[b] = y_train - X_train[:, models[b]] @ coefficients[b]

#         # Evaluate the model on validation data
#         mse = min([mean_squared_error(y_val, apply_rfs(X_train, y_train, X_val, models, k, fit_intercept))
#                    for k in range(1, max_k + 1)])
#         new_k = np.argmin([mean_squared_error(y_val, apply_rfs(X_train, y_train, X_val, models, k, fit_intercept))
#                    for k in range(1, max_k + 1)])

#         # mse = mean_squared_error(y_val, apply_rfs(X_train, y_train, X_val, models, max_k))

#         if mse < best_mse:
#             best_mse = mse
#             best_m = m
#             best_k = new_k
#             best_models = models

#     return best_models, best_m, best_k

# def apply_rfs(X_train, y_train, X_test, models, k, fit_intercept):
#     predictions = np.zeros(len(X_test))
#     for model in models:
#         X_train_subset = X_train[:, model[:k]]
#         X_test_subset = X_test[:, model[:k]]

#         if fit_intercept:
#             # Add a column of ones for the intercept
#             X_train_subset = np.column_stack([np.ones(X_train_subset.shape[0]), X_train_subset])
#             X_test_subset = np.column_stack([np.ones(X_test_subset.shape[0]), X_test_subset])

#         # Fit the model
#         lr = np.linalg.lstsq(X_train_subset, y_train, rcond=None)[0]

#         # Make predictions
#         predictions += X_test_subset @ lr

#     return predictions / len(models)

def run_single_replication(sim, X_std, y, n_features):
    random.seed(123+sim)
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_std, y, test_size=0.7, random_state=sim)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=sim)

    # Lasso
    lasso = Lasso(fit_intercept=True)
    alphas = np.logspace(-4, 1, 50)
    scores = [cross_val_score(Lasso(alpha=alpha), X_train, y_train, cv=5).mean()
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    lasso.set_params(alpha=best_alpha)
    lasso.fit(X_train, y_train)
    mse_lasso= mean_squared_error(y_test, lasso.predict(X_test))

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
    alphas = np.logspace(-4, 1, 50)
    scores = [cross_val_score(Ridge(alpha=alpha), X_train, y_train, cv=5).mean()
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    ridge.set_params(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))

    # RFS without optimized m
    rfs_models_unopt, best_m_unopt, best_k_unopt = optimized_rfs(X_train, y_train, X_val, y_val, n_features, B=500, optimize_m=False, fit_intercept=True)
    rfs_pred_unopt = apply_rfs(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), rfs_models_unopt, best_k_unopt, fit_intercept=True)
    mse_rfs_unopt = mean_squared_error(y_test, rfs_pred_unopt.cpu().numpy())

    # RFS with optimized m
    rfs_models_opt, best_m_opt, best_k_opt = optimized_rfs(X_train, y_train, X_val, y_val, n_features, B=500, optimize_m=True, fit_intercept=True)
    rfs_pred_opt = apply_rfs(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), rfs_models_opt, best_k_opt, fit_intercept=True)
    mse_rfs_opt = mean_squared_error(y_test, rfs_pred_opt.cpu().numpy())

    # FS (RFS with B=1)
    fs_models, _, best_k_fs = optimized_rfs(X_train, y_train, X_val, y_val, n_features, B=1, fs=True)
    fs_pred = apply_rfs(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), fs_models, best_k_fs, fit_intercept=True)
    mse_fs = mean_squared_error(y_test, fs_pred.cpu().numpy())

    # OLS
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train, y_train)
    mse_ols = mean_squared_error(y_test, ols.predict(X_test))

    return (mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, mse_fs, mse_ols, 
            best_m_opt, best_k_unopt, best_k_opt, best_k_fs)

def run_regression_comparison(X, y, label, n_simulations=50):
    warnings.filterwarnings('ignore')

    n_features = X.shape[1]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Initialize multiprocessing pool
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(
                partial(run_single_replication, X_std=X_std, y=y, n_features=n_features),
                range(n_simulations)
            ),
            total=n_simulations,
            desc=f"Running Dataset {label}"
        ))

    # Unpack results
    mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, mse_fs, mse_ols, best_ms, best_ks_unopt, best_ks_opt, best_ks_fs = zip(*results)

    # Convert to numpy arrays
    return (np.array(mse_lasso), np.array(mse_elastic), np.array(mse_ridge), 
            np.array(mse_rfs_unopt), np.array(mse_rfs_opt), np.array(mse_fs), 
            np.array(mse_ols), np.array(best_ms), np.array(best_ks_unopt), 
            np.array(best_ks_opt), np.array(best_ks_fs))

labels = ['Satellite Image']

# 'Political',
# , 'Satellite Image'

data = {}

# Returns a pandas DataFrame
# data['Auto Pricing'] = fetch_data('207_autoPrice') # 159 x 15

# data['Satellite Image'] = fetch_data('294_satellite_image') # 6435 x 35

# data['Political'] = fetch_data('201_pol') # 15000 x 20

# data['Bodyfat'] = fetch_data('560_bodyfat') # 252 x 15

data['Auto Pricing'] = pd.read_csv('207_autoPrice.tsv', sep='\t') # 159 x 15

data['Satellite Image'] = pd.read_csv('294_satellite_image.tsv', sep='\t') # 6435 x 35

data['Political'] = pd.read_csv('201_pol.tsv', sep='\t') # 15000 x 20

data['Bodyfat'] = pd.read_csv('560_bodyfat.tsv', sep='\t') # 252 x 15

# Separate features and target
Xs = {}
ys = {}

for label in labels:
    Xs[label] = data[label].drop('target', axis=1).values
    ys[label] = data[label]['target'].values

# Run on data
results = {}

for label in labels:
    results[label] = run_regression_comparison(Xs[label], ys[label], label, n_simulations=50)

mse_data = []
methods = ['Lasso', 'Elastic Net', 'Ridge', 'RFS Unopt', 'RFS Opt', 'FS', 'OLS']

for label, body in results.items():
    for method, mse_values in zip(methods, body[:7]):  # Only the first 7 elements are MSE values
        for mse in mse_values:
            mse_data.append({
                'Dataset': label,
                'Method': method,
                'MSE': mse
            })
        

df = pd.DataFrame(mse_data)
df.to_csv('real_data_mse.csv', index=False)

best_ms = []

for label, body in results.items():
    for m in body[7]:
        best_ms.append({
                'Dataset': label,
                'Best m': m
            })
        
df_m = pd.DataFrame(best_ms)
df_m.to_csv('real_data_best_ms.csv', index=False)

best_ks = []

for label, body in results.items():
    for method, k_values in zip(methods[3:6], body[8:]): 
        for k in k_values:
            best_ks.append({
                'Dataset': label,
                'Method': method,
                'Best k': k
            })
        

df_k = pd.DataFrame(best_ks)
df_k.to_csv('real_data_best_ks.csv', index=False)