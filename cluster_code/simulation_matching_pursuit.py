import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing
from functools import partial
import warnings

def optimized_rfs(X_train, y_train, X_val, y_val, max_k, B, optimize_m=True, fit_intercept=True, fs=False):
    n, p = X_train.shape
    best_mse = float('inf')
    best_final_model = None
    best_k = None

    m_range = range(1, p + 1) if optimize_m else [p // 3]
    m_range = [p] if fs else m_range

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for m in m_range:
            partial_rfs = partial(perform_single_rfs, X_train=X_train, y_train=y_train, 
                                  max_k=max_k, m=m, p=p)
            
            futures = [executor.submit(partial_rfs, b) for b in range(B)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            models, coefficients = zip(*results)

            final_model, mse_values = apply_rfs(X_train, y_train, X_val, y_val, models, coefficients, max_k, fit_intercept)
            
            best_k_for_m = np.argmin(mse_values) + 1
            mse = mse_values[best_k_for_m - 1]

            if mse < best_mse:
                best_mse = mse
                best_m = m
                best_k = best_k_for_m
                best_final_model = final_model[:, :best_k_for_m]

    return best_final_model, best_m, best_k

def perform_single_rfs(b, X_train, y_train, max_k, m, p):
    model = []
    residual = y_train.copy()
    coefficients = np.zeros((p, max_k))

    for k in range(max_k):
        candidates = np.random.choice([i for i in range(p) if i not in model],
                                      size=min(m, p-len(model)), replace=False)
        inner_products = np.abs(X_train[:, candidates].T @ residual)
        best_feature = candidates[np.argmax(inner_products)]

        model.append(best_feature)
        X_new = X_train[:, model]
        new_coefficients = np.linalg.lstsq(X_new, y_train, rcond=None)[0]
        coefficients[model, k] = new_coefficients
        residual = y_train - X_new @ new_coefficients

    return model, coefficients

def apply_rfs(X_train, y_train, X_val, y_val, models, coefficients, max_k, fit_intercept):
    n_val, p = X_val.shape
    final_model = np.zeros((p, max_k))
    mse_values = []

    for k in range(1, max_k + 1):
        predictions = np.zeros(n_val)
        for model, coef in zip(models, coefficients):
            X_val_subset = X_val[:, model[:k]]
            coef_subset = coef[model[:k], k-1]
            
            if fit_intercept:
                X_val_subset = np.column_stack([np.ones(n_val), X_val_subset])
                coef_subset = np.insert(coef_subset, 0, 0)  # Add intercept coefficient

            predictions += X_val_subset @ coef_subset

        predictions /= len(models)
        mse = mean_squared_error(y_val, predictions)
        mse_values.append(mse)

        # Update final model
        for model, coef in zip(models, coefficients):
            final_model[model[:k], k-1] += coef[model[:k], k-1]
        final_model[:, k-1] /= len(models)

    return final_model, mse_values

def generate_example1(n_train=50, n_val=50, n_test=300, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 8
    beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    sigma = 3

    # Generate correlation matrix
    corr_matrix = np.zeros((n_predictors, n_predictors))
    for i in range(n_predictors):
        for j in range(n_predictors):
            corr_matrix[i, j] = 0.5**abs(i-j)

    # Generate X with the correct covariance structure
    X = np.random.multivariate_normal(np.zeros(n_predictors), corr_matrix, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

def generate_example2(n_train=50, n_val=50, n_test=300, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 8
    beta = np.array([0.85] * 8)
    sigma = 3

    # Use the same correlation structure as in example 1
    corr_matrix = np.zeros((n_predictors, n_predictors))
    for i in range(n_predictors):
        for j in range(n_predictors):
            corr_matrix[i, j] = 0.5**abs(i-j)

    # Generate X with the correct covariance structure
    X = np.random.multivariate_normal(np.zeros(n_predictors), corr_matrix, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

def generate_example3(n_train=100, n_val=100, n_test=400, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 40
    beta = np.concatenate([np.zeros(10), np.full(10, 2), np.zeros(10), np.full(10, 2)])
    sigma = 15

    # Generate correlation matrix with constant correlation of 0.5
    corr_matrix = 0.5 * np.ones((n_predictors, n_predictors)) + 0.5 * np.eye(n_predictors)

    # Generate X with the correct covariance structure
    X = np.random.multivariate_normal(np.zeros(n_predictors), corr_matrix, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

def generate_example4(n_train=50, n_val=50, n_test=400, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 40
    beta = np.concatenate([np.full(15, 3), np.zeros(25)])
    sigma = 15

    # Generate Z variables
    Z1 = np.random.normal(0, 1, n_total)
    Z2 = np.random.normal(0, 1, n_total)
    Z3 = np.random.normal(0, 1, n_total)

    # Generate X
    X = np.zeros((n_total, n_predictors))
    for i in range(5):
        X[:, i] = Z1 + np.random.normal(0, 0.01, n_total)
    for i in range(5, 10):
        X[:, i] = Z2 + np.random.normal(0, 0.01, n_total)
    for i in range(10, 15):
        X[:, i] = Z3 + np.random.normal(0, 0.01, n_total)
    for i in range(15, 40):
        X[:, i] = np.random.normal(0, 0.01, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

def generate_example5(n_train=50, n_val=50, n_test=300, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 100
    beta = np.concatenate([np.full(5, 5), np.zeros(95)])
    sigma = 3

    # Generate correlation matrix with constant correlation of 0.5
    corr_matrix = 0.5 * np.ones((n_predictors, n_predictors)) + 0.5 * np.eye(n_predictors)

    # Generate X with the correct covariance structure
    X = np.random.multivariate_normal(np.zeros(n_predictors), corr_matrix, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

def generate_example6(n_train=50, n_val=50, n_test=300, seed=123):
    np.random.seed(seed)
    n_total = n_train + n_val + n_test
    n_predictors = 100
    beta = np.concatenate([np.full(5, 5), np.zeros(95)])
    sigma = 3

    # Generate correlation matrix with constant correlation of 0
    corr_matrix = np.eye(n_predictors)

    # Generate X with the correct covariance structure
    X = np.random.multivariate_normal(np.zeros(n_predictors), corr_matrix, n_total)

    # Generate response
    y = X @ beta + np.random.normal(0, sigma, n_total)

    # Split the data
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, n_predictors

# Function to run simulations for a single example
def run_single_simulation(sim, generate_example_func, B1=500, B2=500):
    # Generate data
    X_train, y_train, X_val, y_val, X_test, y_test, p = generate_example_func(seed=123+sim)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

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
    alphas = np.logspace(-4, 1, 50)
    scores = [cross_val_score(Ridge(alpha=alpha), X_train, y_train, cv=5).mean()
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    ridge.set_params(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))

    # RFS without optimized m
    rfs_models_unopt, _, best_k_unopt = optimized_rfs(X_train, y_train, X_val, y_val, p, B1, optimize_m=False)
    rfs_pred_unopt = apply_rfs(X_train, y_train, X_test, rfs_models_unopt, best_k_unopt, fit_intercept=True)
    mse_rfs_unopt = mean_squared_error(y_test, rfs_pred_unopt)
    
    # RFS with optimized m
    rfs_models_opt, best_m_opt, best_k_opt = optimized_rfs(X_train, y_train, X_val, y_val, p, B2, optimize_m=True)
    rfs_pred_opt = apply_rfs(X_train, y_train, X_test, rfs_models_opt, best_k_opt, fit_intercept=True)
    mse_rfs_opt = mean_squared_error(y_test, rfs_pred_opt)

    # FS (RFS with B=1)
    fs_models, _, best_k_fs = optimized_rfs(X_train, y_train, X_val, y_val, p, B=1, fs=True)
    fs_pred = apply_rfs(X_train, y_train, X_test, fs_models, best_k_fs, fit_intercept=True)
    mse_fs = mean_squared_error(y_test, fs_pred)

    # OLS
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train, y_train)
    mse_ols = mean_squared_error(y_test, ols.predict(X_test))

    return (mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, mse_fs, mse_ols, 
            best_m_opt, best_k_unopt, best_k_opt, best_k_fs)

def run_simulation(generate_example_name, example_name, n_simulations, B1=500, B2=500):
    generate_example_func = globals()[generate_example_name]
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(
                partial(run_single_simulation, 
                        generate_example_func=generate_example_func, 
                        B1=B1, 
                        B2=B2),
                range(n_simulations)
            ),
            total=n_simulations,
            desc=f"Running {example_name}"
        ))

    # Unpack results
    mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, mse_fs, mse_ols, best_ms, best_ks_unopt, best_ks_opt, best_ks_fs = zip(*results)

    # Convert to numpy arrays
    return (np.array(mse_lasso), np.array(mse_elastic), np.array(mse_ridge), 
            np.array(mse_rfs_unopt), np.array(mse_rfs_opt), np.array(mse_fs), 
            np.array(mse_ols), np.array(best_ms), np.array(best_ks_unopt), 
            np.array(best_ks_opt), np.array(best_ks_fs))

def plot_results(results):
    # Prepare data for plotting
    plot_data = []
    methods = ['Lasso', 'Elastic Net', 'Ridge', 'RFS Unopt', 'RFS Opt', 'FS', 'OLS']

    for example_name, data in results.items():
        for method, mse_values in zip(methods, data[:7]):  # Only the first 6 elements are MSE values
            for mse in mse_values:
                plot_data.append({
                    'Example': example_name,
                    'Method': method,
                    'MSE': mse
                })

    df = pd.DataFrame(plot_data)
    df_chopped = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for example in df.Example.unique():
        for method in df.Method.unique():
            filter = np.where((df['Example']==example) & (df['Method']==method))
            df_chopped.loc[filter]['MSE'] = df_chopped.loc[filter]['MSE'].clip(df_chopped.loc[filter]['MSE'].quantile(0.1), df_chopped.loc[filter]['MSE'].quantile(0.9))

    # Set up the plot
    fig, axes = plt.subplots(2,3, figsize=(30, 20))
    plt.rcParams.update({'font.size': 24})
    axes = axes.flatten()  # Flatten the 2x2 array to make it easier to iterate
    sns.set_style("whitegrid")
    sns.set_palette("Set3")

    for i, (example_name, ax) in enumerate(zip(results.keys(), axes)):
        # Filter data for this example
        example_data = df_chopped[df_chopped['Example'] == example_name]

        # Create the boxplot for this example
        sns.boxplot(x='Method', y='MSE', data=example_data, ax=ax, showfliers=False)

        # Customize the subplot
        ax.set_title(f'{example_name}', fontsize=24)
        ax.set_xlabel('Method', fontsize=24)
        ax.set_ylabel('Mean Squared Error', fontsize=24)
        # ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)

        # Only show y-axis label for left subplots
        if i % 3 != 0:
            ax.set_ylabel('')

        # Only show x-axis label for bottom subplots
        if i < 3:
            ax.set_xlabel('')

    # Add a main title
    fig.suptitle('Comparison of Regression Methods across Examples', fontsize=35)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust top spacing for main title
    plt.savefig('simulations.png')
    plt.show()

warnings.filterwarnings('ignore')

# Simulation parameters
n_simulations = 50

# Run simulations for all examples
examples = [
    ("generate_example1", "Example 1"),
    ("generate_example2", "Example 2"),
    ("generate_example3", "Example 3"),
    ("generate_example4", "Example 4"),
    ("generate_example5", "Example 5"),
    ("generate_example6", "Example 6")
]

results = {}

for i in range(len(examples)):
    example_name = examples[i][1]
    generate_func = examples[i][0]
    results[example_name] = run_simulation(generate_func, example_name, n_simulations)

# Print results
for example_name, (mse_lasso, mse_elastic, mse_ridge, mse_rfs_unopt, mse_rfs_opt, mse_fs, mse_ols, best_ms, best_ks_unopt, best_ks_opt, best_ks_fs) in results.items():
    print(f"\nResults for {example_name}:")
    print(f"Median MSE Lasso: {np.median(mse_lasso):.3f} (±{np.std(mse_lasso, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE Elastic Net: {np.median(mse_elastic):.3f} (±{np.std(mse_elastic, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE Ridge: {np.median(mse_ridge):.3f} (±{np.std(mse_ridge, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE RFS Unoptimized: {np.median(mse_rfs_unopt):.3f} (±{np.std(mse_rfs_unopt, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE RFS Optimized: {np.median(mse_rfs_opt):.3f} (±{np.std(mse_rfs_opt, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE FS: {np.median(mse_fs):.3f} (±{np.std(mse_fs, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median MSE OLS: {np.median(mse_ols):.3f} (±{np.std(mse_ols, ddof=1)/np.sqrt(n_simulations):.3f})")
    print(f"Median best m: {np.median(best_ms):.2f}")
    print(f"Median best k unoptimized: {np.median(best_ks_unopt):.2f}")
    print(f"Median best k optimized: {np.median(best_ks_opt):.2f}")
    print(f"Median best k FS: {np.median(best_ks_fs):.2f}")

# Assuming 'results' is your dictionary containing the simulation results
plot_results(results)

mse_data = []
methods = ['Lasso', 'Elastic Net', 'Ridge', 'RFS Unopt', 'RFS Opt', 'FS', 'OLS']

for example_name, data in results.items():
    for method, mse_values in zip(methods, data[:7]):  # Only the first 7 elements are MSE values
        for mse in mse_values:
            mse_data.append({
                'Example': example_name,
                'Method': method,
                'MSE': mse
            })
        

df = pd.DataFrame(mse_data)
df.to_csv('simulations_mse.csv', index=False)

best_ms = []

for example_name, data in results.items():
    for m in data[7]:
        best_ms.append({
                'Example': example_name,
                'Best m': m
            })
        
df_m = pd.DataFrame(best_ms)
df_m.to_csv('best_ms.csv', index=False)

best_ks = []

for example_name, data in results.items():
    for method, k_values in zip(methods[3:6], data[8:]): 
        for k in k_values:
            best_ks.append({
                'Example': example_name,
                'Method': method,
                'Best k': k
            })
        

df_k = pd.DataFrame(best_ks)
df_k.to_csv('best_ks.csv', index=False)