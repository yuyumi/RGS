import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_available_methods(df, metric_prefix):
    """Get list of available methods based on column prefixes."""
    methods = ['lasso', 'ridge', 'elastic', 'gs', 'rgs']
    return [m for m in methods if f'{metric_prefix}_{m}' in df.columns]

def plot_metric_by_sigma(results_path, metric='mse', save_path=None):
    """Generic plotting function for any metric vs sigma."""
    df = pd.read_csv(results_path)
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors for different methods
    colors = ['#2ca02c', '#ff7f7f', '#17becf', '#e6b3e6', '#000000']
    
    # Find available methods
    available_methods = get_available_methods(df, metric)
    
    if not available_methods:
        print(f"Warning: No {metric} data found for any method")
        plt.close(fig)
        return None
    
    # Plot each available method
    for i, method in enumerate(available_methods):
        metric_col = f'{metric}_{method}'
        mean_val = df.groupby('sigma')[metric_col].mean()
        ax.plot(mean_val.index, mean_val.values,
               marker='o',
               color=colors[i],
               label=method.upper())
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Sigma (Noise Level)', fontsize=12)
    if (metric == 'mse'):
        metric_label = 'Mean Square Error'
    elif (metric == 'insample'):
        metric_label = 'In-sample Error'
    else:
        metric_label = 'Degrees of Freedom'
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} by Sigma Level', fontsize=14)
    
    if available_methods:
        ax.legend(loc='upper left', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_metric_by_variance_explained(results_path, metric='mse', save_path=None):
    """Generic plotting function for any metric vs PVE."""
    df = pd.read_csv(results_path)
    
    # Calculate variance explained (PVE)
    norm_beta = 10  # Default value
    df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors for different methods
    colors = ['#2ca02c', '#ff7f7f', '#17becf', '#e6b3e6', '#000000']
    
    # Find available methods
    available_methods = get_available_methods(df, metric)
    
    if not available_methods:
        print(f"Warning: No {metric} data found for any method")
        plt.close(fig)
        return None
    
    # Plot each available method
    for i, method in enumerate(available_methods):
        metric_col = f'{metric}_{method}'
        mean_val = df.groupby('var_explained')[metric_col].mean()
        ax.plot(mean_val.index, mean_val.values,
               marker='o',
               color=colors[i],
               label=method.upper())
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Proportion of Variance Explained (PVE)', fontsize=12)
    if (metric == 'mse'):
        metric_label = 'Mean Square Error'
    elif (metric == 'insample'):
        metric_label = 'In-sample Error'
    else:
        metric_label = 'Degrees of Freedom'
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} by PVE', fontsize=14)
    
    if available_methods:
        ax.legend(loc='upper right' if ((metric == 'mse') or (metric == 'insample')) else 'upper left', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_metric_vs_k(results_path, target_sigma, metric='mse', save_path=None):
    """Generic plotting function for any metric vs k."""
    df = pd.read_csv(results_path)
    df_sigma = df[np.abs(df['sigma'] - target_sigma) < 1e-6]
    
    if df_sigma.empty:
        print(f"Warning: No data found for sigma = {target_sigma}")
        return None
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors for different methods
    colors = ['#2ca02c', '#ff7f7f', '#17becf', '#e6b3e6', '#000000']
    
    # Find available methods
    available_methods = get_available_methods(df_sigma, metric)
    baseline_methods = [m for m in ['lasso', 'ridge', 'elastic'] if m in available_methods]
    advanced_methods = [m for m in ['gs', 'rgs'] if m in available_methods]
    
    if not available_methods:
        print(f"Warning: No {metric} data found for any method")
        plt.close(fig)
        return None
    
    # Plot baselines
    for i, method in enumerate(baseline_methods):
        metric_col = f'{metric}_{method}'
        metric_value = df_sigma[metric_col].mean()
        ax.axhline(y=metric_value, color=colors[i], linestyle='--', 
                  label=f'{method.upper()}')
    
    # Plot k-dependent methods if k values are available
    if 'best_k' in df_sigma.columns:
        k_values = sorted(df_sigma['best_k'].unique())
        
        for method in advanced_methods:
            metric_col = f'{metric}_{method}'
            method_values = [df_sigma[df_sigma['best_k'] == k][metric_col].mean() 
                           for k in k_values]
            method_idx = ['gs', 'rgs'].index(method) + 3  # Offset for baseline colors
            ax.plot(k_values, method_values, marker='o', 
                   color=colors[method_idx], label=method.upper())
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('k', fontsize=12)
    if (metric == 'mse'):
        metric_label = 'Mean Square Error'
    elif (metric == 'insample'):
        metric_label = 'In-sample Error'
    else:
        metric_label = 'Degrees of Freedom'
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_label} vs k (σ = {target_sigma:.2f})', fontsize=14)
    
    if available_methods:
        ax.legend(loc='upper right', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

# Convenience functions that don't pass the metric parameter
def plot_mse_by_sigma(results_path, save_path=None):
    return plot_metric_by_sigma(results_path, metric='mse', save_path=save_path)

def plot_insample_by_sigma(results_path, save_path=None):
    return plot_metric_by_sigma(results_path, metric='insample', save_path=save_path)

def plot_df_by_sigma(results_path, save_path=None):
    return plot_metric_by_sigma(results_path, metric='df', save_path=save_path)

def plot_mse_by_variance_explained(results_path, save_path=None):
    return plot_metric_by_variance_explained(results_path, metric='mse', save_path=save_path)

def plot_insample_by_variance_explained(results_path, save_path=None):
    return plot_metric_by_variance_explained(results_path, metric='insample', save_path=save_path)

def plot_df_by_variance_explained(results_path, save_path=None):
    return plot_metric_by_variance_explained(results_path, metric='df', save_path=save_path)

def plot_mse_vs_k(results_path, target_sigma, save_path=None):
    return plot_metric_vs_k(results_path, target_sigma, metric='mse', save_path=save_path)

def plot_df_vs_k(results_path, target_sigma, save_path=None):
    return plot_metric_vs_k(results_path, target_sigma, metric='df', save_path=save_path)