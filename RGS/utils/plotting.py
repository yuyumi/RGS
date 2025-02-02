import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

def plot_mse_by_sigma(csv_path, save_path=None):
    """Plot MSE vs sigma for different methods, using optimal configurations"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get parameters for each noise level
    noise_levels = sorted(df['noise_level'].unique(), key=lambda x: float(x.split('_')[1]))
    optimal_gs_mse = []
    optimal_rgs_mse = []
    
    # For each noise level, get the optimal MSE values
    for noise_level in noise_levels:
        noise_data = df[df['noise_level'] == noise_level]
        
        # Find best GS k value based on penalized
        gs_mse_cols = [col for col in noise_data.columns if col.startswith('pen_gs_k')]
        best_gs_col = min(gs_mse_cols, key=lambda x: noise_data[x].mean())
        best_gs_k = int(best_gs_col.replace('pen_gs_k', ''))
        best_gs_mse = noise_data[f'mse_gs_k{best_gs_k}'].mean()
        optimal_gs_mse.append(best_gs_mse)
        
        # Find best RGS m and k values based on penalized
        rgs_mse_cols = [col for col in noise_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_mse_cols, key=lambda x: noise_data[x].mean())
        col_parts = best_rgs_col.replace('pen_rgs_m', '').split('_k')
        m_val = int(col_parts[0])
        k_val = int(col_parts[1])
        best_rgs_mse = noise_data[f'mse_rgs_m{m_val}_k{k_val}'].mean()
        optimal_rgs_mse.append(best_rgs_mse)
    
    # Create plotting DataFrame
    plot_df = pd.DataFrame({
        'sigma': [float(level.split('_')[1]) for level in noise_levels],
        'mse_gs_optimal': optimal_gs_mse,
        'mse_rgs_optimal': optimal_rgs_mse,
        # Fix the ordering of baseline methods to match noise_levels
        'mse_lasso': [df[df['noise_level'] == level]['mse_lasso'].mean() for level in noise_levels],
        'mse_ridge': [df[df['noise_level'] == level]['mse_ridge'].mean() for level in noise_levels],
        'mse_elastic': [df[df['noise_level'] == level]['mse_elastic'].mean() for level in noise_levels]
    })
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors matching the provided graph scheme with light purple and black
    colors = ['#2ca02c', '#ff7f7f', '#17becf', '#e6b3e6', '#000000']
    
    # Plot each method
    mse_columns = ['mse_lasso', 'mse_ridge', 'mse_elastic', 'mse_gs_optimal', 'mse_rgs_optimal']
    
    for i, mse_col in enumerate(mse_columns):
        label = mse_col.replace('mse_', '').replace('_optimal', '').upper()
        ax.plot(plot_df['sigma'], plot_df[mse_col],
                marker='o',
                color=colors[i],
                label=label)
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Sigma (Noise Level)', fontsize=12)
    ax.set_ylabel('Mean Square Error', fontsize=12)
    ax.set_title('Mean Square Error by Sigma Level\n(Optimal configurations for GS and RGS)', 
                fontsize=14)
    
    # Add legend inside the plot
    ax.legend(loc='upper left', 
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_mse_by_variance_explained(csv_path, norm_beta=10, save_path=None):
    """Plot MSE vs proportion of variance explained for different methods"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get parameters for each noise level
    noise_levels = sorted(df['noise_level'].unique(), key=lambda x: float(x.split('_')[1]))
    optimal_gs_mse = []
    optimal_rgs_mse = []
    
    # For each noise level, get the optimal MSE values
    for noise_level in noise_levels:
        noise_data = df[df['noise_level'] == noise_level]
        
        # Find best GS k value based on penalized
        gs_mse_cols = [col for col in noise_data.columns if col.startswith('pen_gs_k')]
        best_gs_col = min(gs_mse_cols, key=lambda x: noise_data[x].mean())
        best_gs_k = int(best_gs_col.replace('pen_gs_k', ''))
        best_gs_mse = noise_data[f'mse_gs_k{best_gs_k}'].mean()
        optimal_gs_mse.append(best_gs_mse)
        
        # Find best RGS m and k values based on penalized
        rgs_mse_cols = [col for col in noise_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_mse_cols, key=lambda x: noise_data[x].mean())
        col_parts = best_rgs_col.replace('pen_rgs_m', '').split('_k')
        m_val = int(col_parts[0])
        k_val = int(col_parts[1])
        best_rgs_mse = noise_data[f'mse_rgs_m{m_val}_k{k_val}'].mean()
        optimal_rgs_mse.append(best_rgs_mse)
    
    # Create plotting DataFrame
    plot_df = pd.DataFrame({
        'sigma': [float(level.split('_')[1]) for level in noise_levels],
        'mse_gs_optimal': optimal_gs_mse,
        'mse_rgs_optimal': optimal_rgs_mse,
        # Fix the ordering of baseline methods to match noise_levels
        'mse_lasso': [df[df['noise_level'] == level]['mse_lasso'].mean() for level in noise_levels],
        'mse_ridge': [df[df['noise_level'] == level]['mse_ridge'].mean() for level in noise_levels],
        'mse_elastic': [df[df['noise_level'] == level]['mse_elastic'].mean() for level in noise_levels]
    })
    plot_df['var_explained'] = norm_beta / (norm_beta + plot_df['sigma']**2)
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors optimized for both color and B&W printing
    colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2']
    
    # Plot each method
    mse_columns = ['mse_lasso', 'mse_ridge', 'mse_elastic', 'mse_gs_optimal', 'mse_rgs_optimal']
    
    for i, mse_col in enumerate(mse_columns):
        label = mse_col.replace('mse_', '').replace('_optimal', '').upper()
        ax.plot(plot_df['var_explained'], plot_df[mse_col],
                marker='o',
                color=colors[i],
                label=label)
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('PVE', fontsize=12)
    ax.set_ylabel('In-sample Error', fontsize=12)
    ax.set_title('In-sample Error by Proportion of Variance Explained', 
                fontsize=14)
    
    # Add legend inside the plot
    ax.legend(loc='upper right', 
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_df_by_sigma(csv_path, save_path=None):
    """Plot degrees of freedom vs sigma for different methods"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get parameters for each noise level
    noise_levels = sorted(df['noise_level'].unique(), key=lambda x: float(x.split('_')[1]))
    optimal_gs_df = []
    optimal_rgs_df = []
    
    # For each noise level, get the optimal df values
    for noise_level in noise_levels:
        noise_data = df[df['noise_level'] == noise_level]
        
        # Find best GS k value based on penalized
        gs_cols = [col for col in noise_data.columns if col.startswith('pen_gs_k')]
        best_gs_col = min(gs_cols, key=lambda x: noise_data[x].mean())
        best_gs_k = int(best_gs_col.replace('pen_gs_k', ''))
        best_gs_df = noise_data[f'df_gs_k{best_gs_k}'].mean()
        optimal_gs_df.append(best_gs_df)
        
        # Find best RGS m and k values based on penalized
        rgs_cols = [col for col in noise_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_cols, key=lambda x: noise_data[x].mean())
        col_parts = best_rgs_col.replace('pen_rgs_m', '').split('_k')
        m_val = int(col_parts[0])
        k_val = int(col_parts[1])
        best_rgs_df = noise_data[f'df_rgs_m{m_val}_k{k_val}'].mean()
        optimal_rgs_df.append(best_rgs_df)
    
    # Create plotting DataFrame
    plot_df = pd.DataFrame({
        'sigma': [float(level.split('_')[1]) for level in noise_levels],
        'df_gs_optimal': optimal_gs_df,
        'df_rgs_optimal': optimal_rgs_df,
        # Fix the ordering of baseline methods to match noise_levels
        'df_lasso': [df[df['noise_level'] == level]['df_lasso'].mean() for level in noise_levels],
        'df_ridge': [df[df['noise_level'] == level]['df_ridge'].mean() for level in noise_levels],
        'df_elastic': [df[df['noise_level'] == level]['df_elastic'].mean() for level in noise_levels]
    })
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors optimized for both color and B&W printing
    colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2']
    
    # Plot each method
    df_columns = ['df_lasso', 'df_ridge', 'df_elastic', 'df_gs_optimal', 'df_rgs_optimal']
    
    for i, df_col in enumerate(df_columns):
        label = df_col.replace('df_', '').replace('_optimal', '').upper()
        ax.plot(plot_df['sigma'], plot_df[df_col],
                marker='o',
                color=colors[i],
                label=label)
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Sigma (Noise Level)', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title('Degrees of Freedom by Sigma Level\n(Optimal configurations for GS and RGS)', 
                fontsize=14)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), 
             loc='upper left', 
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_df_by_variance_explained(csv_path, norm_beta=10, save_path=None):
    """Plot degrees of freedom vs proportion of variance explained"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get parameters for each noise level
    noise_levels = sorted(df['noise_level'].unique(), key=lambda x: float(x.split('_')[1]))
    optimal_gs_df = []
    optimal_rgs_df = []
    
    # For each noise level, get the optimal df values
    for noise_level in noise_levels:
        noise_data = df[df['noise_level'] == noise_level]
        
        # Find best GS k value based on penalized
        gs_cols = [col for col in noise_data.columns if col.startswith('pen_gs_k')]
        best_gs_col = min(gs_cols, key=lambda x: noise_data[x].mean())
        best_gs_k = int(best_gs_col.replace('pen_gs_k', ''))
        best_gs_df = noise_data[f'df_gs_k{best_gs_k}'].mean()
        optimal_gs_df.append(best_gs_df)
        
        # Find best RGS m and k values based on penalized
        rgs_cols = [col for col in noise_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_cols, key=lambda x: noise_data[x].mean())
        col_parts = best_rgs_col.replace('pen_rgs_m', '').split('_k')
        m_val = int(col_parts[0])
        k_val = int(col_parts[1])
        best_rgs_df = noise_data[f'df_rgs_m{m_val}_k{k_val}'].mean()
        optimal_rgs_df.append(best_rgs_df)
    
    # Create plotting DataFrame with variance explained
    plot_df = pd.DataFrame({
        'sigma': [float(level.split('_')[1]) for level in noise_levels],
        'df_gs_optimal': optimal_gs_df,
        'df_rgs_optimal': optimal_rgs_df,
        # Fix the ordering of baseline methods to match noise_levels
        'df_lasso': [df[df['noise_level'] == level]['df_lasso'].mean() for level in noise_levels],
        'df_ridge': [df[df['noise_level'] == level]['df_ridge'].mean() for level in noise_levels],
        'df_elastic': [df[df['noise_level'] == level]['df_elastic'].mean() for level in noise_levels]
    })
    plot_df['var_explained'] = norm_beta / (norm_beta + plot_df['sigma']**2)
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Colors matching the provided graph scheme
    colors = ['#2ca02c', '#ff7f7f', '#17becf', '#98df8a', '#9edae5']
    
    # Plot each method
    df_columns = ['df_lasso', 'df_ridge', 'df_elastic', 'df_gs_optimal', 'df_rgs_optimal']
    
    for i, df_col in enumerate(df_columns):
        label = df_col.replace('df_', '').replace('_optimal', '').upper()
        ax.plot(plot_df['var_explained'], plot_df[df_col],
                marker='o',
                color=colors[i],
                label=label)
    
    # Set log scale and customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Proportion of Variance Explained', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title('Degrees of Freedom by Proportion of Variance Explained', 
                fontsize=14)
    
    # Add legend on right side
    ax.legend(bbox_to_anchor=(1.05, 1),
             loc='upper left',
             fontsize=10)
    
    # Adjust layout to ensure legend fits
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def extract_k_value(column_name):
    """Extract k value from column name"""
    return int(column_name.split('_k')[-1])

def extract_m_value(column_name):
    """Extract m value from column name"""
    return int(column_name.split('_m')[-1].split('_')[0])

# Create a larger color palette for more methods
def get_color_palette(n_colors):
    """Generate a color palette with the required number of colors"""
    # Base colors - colorblind friendly
    base_colors = [
        '#E69F00',   # orange
        '#56B4E9',   # light blue
        '#009E73',   # green
        '#CC79A7',   # pink
        '#0072B2',   # dark blue
        '#D55E00',   # red
        '#882255',   # purple
        '#44AA99',   # teal
        '#F0E442',   # yellow
        '#117733',   # dark green
        '#332288',   # indigo
        '#AA4499',   # violet
        '#88CCEE',   # light cyan
        '#999933',   # olive
        '#CC6677',   # rose
        '#AA4466',   # wine
        '#4477AA',   # steel blue
        '#228833',   # forest green
        '#CCBB44',   # mustard
        '#66CCEE',   # sky blue
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # If we need more colors, use interpolation
    additional_colors = []
    for i in range(n_colors - len(base_colors)):
        idx1 = i % (len(base_colors) - 1)
        idx2 = (idx1 + 1) % len(base_colors)
        t = 0.5  # Mix ratio
        c1 = np.array([int(base_colors[idx1][i:i+2], 16) for i in (1,3,5)])
        c2 = np.array([int(base_colors[idx2][i:i+2], 16) for i in (1,3,5)])
        mixed = t * c1 + (1-t) * c2
        additional_colors.append('#%02x%02x%02x' % tuple(mixed.astype(int)))
    
    return base_colors + additional_colors

def plot_mse_vs_k(csv_path, target_sigma, tolerance=1e-6, save_path=None):
    """Plot MSE vs k curves for GS and RGS methods at a specific sigma value"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows for target sigma
    mask = np.abs(df['sigma'] - target_sigma) < tolerance
    if not mask.any():
        raise ValueError(f"No data found for sigma = {target_sigma}")
    filtered_df = df[mask]
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors optimized for both color and B&W printing
    colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2']
    
    # First get k values from GS columns
    gs_cols = [col for col in df.columns if col.startswith('mse_gs_k')]
    k_values = sorted(list(set([extract_k_value(col) for col in gs_cols])))
    
    # Calculate mean MSE for each k value for GS
    gs_mse_values = []
    for k in k_values:
        col = f'mse_gs_k{k}'
        gs_mse_values.append(filtered_df[col].mean())
    
    # Plot GS
    ax.plot(k_values, gs_mse_values,
            marker='o',
            color=colors[0],
            label='GS')
    
    # Plot RGS for different m values using the same k values
    rgs_base_cols = [col for col in df.columns if col.startswith('mse_rgs_m')]
    unique_m_values = sorted(list(set([extract_m_value(col) for col in rgs_base_cols])))
    unique_m_values = [m for m in unique_m_values if m != 250]  # Exclude m=250
    
    for idx, m in enumerate(unique_m_values, 1):
        # Use the same k values as GS
        rgs_mse_values = []
        valid_k = True
        for k in k_values:
            col = f'mse_rgs_m{m}_k{k}'
            if col in df.columns:
                rgs_mse_values.append(filtered_df[col].mean())
            else:
                valid_k = False
                break
                
        if valid_k:
            ax.plot(k_values, rgs_mse_values,
                    marker='o',
                    color=colors[idx % len(colors)],
                    label=f'RGS (m = {m})')
    
    # Plot baseline methods
    baseline_methods = ['mse_lasso', 'mse_ridge', 'mse_elastic']
    baseline_names = ['LASSO', 'RIDGE', 'ELASTIC']
    
    for idx, (method, name) in enumerate(zip(baseline_methods, baseline_names)):
        mse_value = filtered_df[method].mean()
        ax.axhline(y=mse_value,
                  color=colors[idx],
                  linestyle='--',
                  label=name)
    
    # Customize plot
    ax.set_yscale('log')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Mean Square Error', fontsize=12)
    ax.set_title(f'MSE vs k (σ = {target_sigma})',
                fontsize=14)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1),
             loc='upper left',
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_df_vs_k(csv_path, target_sigma, tolerance=1e-6, save_path=None):
    """Plot df vs k curves for GS and RGS methods at a specific sigma value"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows for target sigma
    mask = np.abs(df['sigma'] - target_sigma) < tolerance
    if not mask.any():
        raise ValueError(f"No data found for sigma = {target_sigma}")
    filtered_df = df[mask]
    
    # Create figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Calculate number of colors needed (GS + number of RGS m values)
    rgs_base_cols = [col for col in df.columns if col.startswith('df_rgs_m')]
    unique_m_values = sorted(list(set([extract_m_value(col) for col in rgs_base_cols])))
    unique_m_values = [m for m in unique_m_values if m != 250]  # Exclude m=250
    n_colors_needed = 1 + len(unique_m_values)  # GS + RGS methods
    
    # Get color palette
    colors = get_color_palette(n_colors_needed)
    
    # First get k values from GS columns
    gs_cols = [col for col in df.columns if col.startswith('df_gs_k')]
    k_values = sorted(list(set([extract_k_value(col) for col in gs_cols])))
    
    # Calculate mean DF for each k value for GS
    gs_df_values = []
    for k in k_values:
        col = f'df_gs_k{k}'
        gs_df_values.append(filtered_df[col].mean())
    
    # Plot GS
    ax.plot(k_values, gs_df_values,
            marker='o',
            color=colors[0],
            label='GS')
    
    # Plot RGS for different m values using the same k values
    rgs_base_cols = [col for col in df.columns if col.startswith('df_rgs_m')]
    unique_m_values = sorted(list(set([extract_m_value(col) for col in rgs_base_cols])))
    unique_m_values = [m for m in unique_m_values if m != 500]  # Exclude m=250
    
    for idx, m in enumerate(unique_m_values, 1):
        # Use the same k values as GS
        rgs_df_values = []
        valid_k = True
        for k in k_values:
            col = f'df_rgs_m{m}_k{k}'
            if col in df.columns:
                rgs_df_values.append(filtered_df[col].mean())
            else:
                valid_k = False
                break
                
        if valid_k:
            ax.plot(k_values, rgs_df_values,
                    marker='o',
                    color=colors[idx % len(colors)],
                    label=f'RGS (m = {m})')
    
    # Customize plot
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title(f'Degrees of Freedom vs k (σ = {target_sigma})',
                fontsize=14)
    
    # Add legend
    ax.legend(
             loc='upper left',
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig