import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

def get_optimal_configuration(df, row_idx, pen_prefix, mse_prefix):
    """Get the optimal MSE value based on minimum penalty score for a given row"""
    pen_cols = [col for col in df.columns if col.startswith(pen_prefix)]
    
    # Get penalty scores for this row
    penalties = {col: df.iloc[row_idx][col] for col in pen_cols}
    
    # Find configuration with minimum penalty
    min_pen_col = min(penalties.items(), key=lambda x: x[1])[0]
    
    # Get corresponding MSE column
    config_suffix = min_pen_col.replace(pen_prefix, '')
    mse_col = mse_prefix + config_suffix
    
    return df.iloc[row_idx][mse_col]

def plot_mse_by_sigma(csv_path, save_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create arrays to store optimal MSE values
    optimal_gs_mse = []
    optimal_rgs_mse = []
    
    # For each row, get optimal configurations
    for idx in range(len(df)):
        optimal_gs_mse.append(get_optimal_configuration(df, idx, 'pen_gs_', 'mse_gs_'))
        optimal_rgs_mse.append(get_optimal_configuration(df, idx, 'pen_rgs_', 'mse_rgs_'))
    
    # Add optimal values to dataframe
    df['mse_gs_optimal'] = optimal_gs_mse
    df['mse_rgs_optimal'] = optimal_rgs_mse
    
    # Get final MSE columns to plot
    mse_columns = ['mse_lasso', 'mse_ridge', 'mse_elastic', 'mse_gs_optimal', 'mse_rgs_optimal']
    
    # Create figure
    sns.set_style("white")  # Remove gridlines
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    # Create color palette - optimized for both color and B&W printing
    colors = ['#E69F00',   # orange
              '#56B4E9',   # light blue
              '#009E73',   # green
              '#CC79A7',   # pink
              '#0072B2']   # dark blue
    
    # Plot each method
    for i, mse_col in enumerate(mse_columns):
        # Calculate statistics for each sigma
        stats_df = df.groupby('sigma').agg({
            mse_col: ['mean', 'std', 'count']
        }).reset_index()
        
        # Extract values from multi-index columns and convert to numpy arrays
        sigma_values = stats_df['sigma'].to_numpy()
        mean_values = stats_df[(mse_col, 'mean')].to_numpy()
        std_values = stats_df[(mse_col, 'std')].to_numpy()
        count_values = stats_df[(mse_col, 'count')].to_numpy()
        
        # Calculate standard error
        se_values = std_values / np.sqrt(count_values)
        
        # Create label
        label = mse_col.replace('mse_', '').replace('_optimal', '').upper()
        
        # Plot mean line with error bars
        ax.errorbar(sigma_values, mean_values,
                    yerr=1.96 * se_values,
                    marker='o',
                    color=colors[i],
                    label=label,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.6,
                    zorder=1)
        # Plot the main line again on top to ensure it's solid
        ax.plot(sigma_values, mean_values,
                marker='o',
                color=colors[i],
                zorder=2)
        
        # Set log scale for y-axis
        plt.yscale('log')

    # Customize plot
    plt.xlabel('Sigma (Noise Level)', fontsize=12)
    plt.ylabel('Mean Square Error', fontsize=12)
    plt.title('Mean Square Error by Sigma Level with 95% Confidence Intervals\n(Optimal configurations for gs and RGS)', fontsize=14)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save the figure if a save path is provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/mse_by_sigma_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_mse_by_variance_explained(csv_path, norm_beta=10, save_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create arrays to store optimal MSE values
    optimal_gs_mse = []
    optimal_rgs_mse = []
    
    # For each row, get optimal configurations
    for idx in range(len(df)):
        optimal_gs_mse.append(get_optimal_configuration(df, idx, 'pen_gs_', 'mse_gs_'))
        optimal_rgs_mse.append(get_optimal_configuration(df, idx, 'pen_rgs_', 'mse_rgs_'))
    
    # Add optimal values to dataframe
    df['mse_gs_optimal'] = optimal_gs_mse
    df['mse_rgs_optimal'] = optimal_rgs_mse
    
    # Calculate proportion of variance explained
    df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
    
    # Get MSE columns to plot
    mse_columns = ['mse_lasso', 'mse_ridge', 'mse_elastic', 'mse_gs_optimal', 'mse_rgs_optimal']
    
    # Create figure
    sns.set_style("white")  # Remove gridlines
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    # Create color palette - optimized for both color and B&W printing
    colors = {
        'mse_lasso': '#E63946',    # Deep Red
        'mse_ridge': '#1D3557',    # Navy Blue
        'mse_elastic': '#2A9D8F',  # Teal
        'mse_gs_optimal': '#F4A261',  # Orange
        'mse_rgs_optimal': '#9B4F96'   # Purple
    }
    
    # Plot each method
    for i, mse_col in enumerate(mse_columns):
        # Calculate statistics for each unique variance explained value
        stats_df = df.groupby('var_explained').agg({
            mse_col: ['mean', 'std', 'count']
        }).reset_index()
        
        # Sort by variance explained for proper line plotting
        stats_df = stats_df.sort_values('var_explained')
        
        # Extract values from multi-index columns and convert to numpy arrays
        var_values = stats_df['var_explained'].to_numpy()
        mean_values = stats_df[(mse_col, 'mean')].to_numpy()
        std_values = stats_df[(mse_col, 'std')].to_numpy()
        count_values = stats_df[(mse_col, 'count')].to_numpy()
        
        # Calculate standard error
        se_values = std_values / np.sqrt(count_values)
        
        # Create label
        label = mse_col.replace('mse_', '').replace('_optimal', '').upper()
        
        # Plot mean line with error bars
        ax.errorbar(var_values, mean_values,
                    yerr=1.96 * se_values,
                    marker='o',
                    color=colors[mse_col],
                    label=label,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.6,
                    zorder=1)
        # Plot the main line again on top to ensure it's solid
        ax.plot(var_values, mean_values,
                marker='o',
                color=colors[mse_col],
                zorder=2)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Customize plot
    ax.set_xlabel('Proportion of Variance Explained', fontsize=12)
    ax.set_ylabel('In-sample Error', fontsize=12)
    ax.set_title(f'In-sample Error by PVE', 
                fontsize=14)
    
    # Add legend in top right corner of plot
    legend = plt.legend(loc='upper right',
                       fontsize=10,
                       frameon=True,
                       edgecolor='black')
    legend.get_frame().set_alpha(1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/mse_by_variance_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_df_by_sigma(csv_path, save_path=None):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create arrays to store optimal df values
    optimal_gs_df = []
    optimal_rgs_df = []
    
    # For each row, get optimal configurations
    for idx in range(len(df)):
        optimal_gs_df.append(get_optimal_configuration(df, idx, 'pen_gs_', 'df_gs_'))
        optimal_rgs_df.append(get_optimal_configuration(df, idx, 'pen_rgs_', 'df_rgs_'))
    
    # Add optimal values to dataframe
    df['df_gs_optimal'] = optimal_gs_df
    df['df_rgs_optimal'] = optimal_rgs_df
    
    # Get df columns to plot
    df_columns = ['df_lasso', 'df_ridge', 'df_elastic', 'df_gs_optimal', 'df_rgs_optimal']
    
    # Create figure
    sns.set_style("white")  # Remove gridlines
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    # Create color palette - optimized for both color and B&W printing
    colors = ['#E69F00',   # orange
              '#56B4E9',   # light blue
              '#009E73',   # green
              '#CC79A7',   # pink
              '#0072B2']   # dark blue
    
    # Plot each method
    for i, df_col in enumerate(df_columns):
        # Calculate statistics for each sigma
        stats_df = df.groupby('sigma').agg({
            df_col: ['mean', 'std', 'count']
        }).reset_index()
        
        # Extract values from multi-index columns and convert to numpy arrays
        sigma_values = stats_df['sigma'].to_numpy()
        mean_values = stats_df[(df_col, 'mean')].to_numpy()
        std_values = stats_df[(df_col, 'std')].to_numpy()
        count_values = stats_df[(df_col, 'count')].to_numpy()
        
        # Calculate standard error
        se_values = std_values / np.sqrt(count_values)
        
        # Create label
        label = df_col.replace('df_', '').replace('_optimal', '').upper()
        
        # Plot mean line with error bars
        ax.errorbar(sigma_values, mean_values,
                    yerr=1.96 * se_values,
                    marker='o',
                    color=colors[i],
                    label=label,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.6,
                    zorder=1)
        # Plot the main line again on top to ensure it's solid
        ax.plot(sigma_values, mean_values,
                marker='o',
                color=colors[i],
                zorder=2)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Customize plot
    ax.set_xlabel('Sigma (Noise Level)', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title('Degrees of Freedom by Sigma Level with 95% Confidence Intervals\n(Optimal configurations for gs and RGS)', 
                fontsize=14)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), 
             loc='upper left', 
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/df_by_sigma_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_df_by_variance_explained(csv_path, norm_beta=10, save_path=None):
    """Plot degrees of freedom against proportion of variance explained"""
    # Generate default save path if none provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/df_by_variance_{timestamp}.png"
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create arrays to store optimal df values
    optimal_gs_df = []
    optimal_rgs_df = []
    
    # For each row, get optimal configurations
    for idx in range(len(df)):
        optimal_gs_df.append(get_optimal_configuration(df, idx, 'pen_gs_', 'df_gs_'))
        optimal_rgs_df.append(get_optimal_configuration(df, idx, 'pen_rgs_', 'df_rgs_'))
    
    # Add optimal values to dataframe
    df['df_gs_optimal'] = optimal_gs_df
    df['df_rgs_optimal'] = optimal_rgs_df
    
    # Calculate proportion of variance explained
    df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
    
    # Get df columns to plot
    df_columns = ['df_lasso', 'df_ridge', 'df_elastic', 'df_gs_optimal', 'df_rgs_optimal']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create color palette
    colors = sns.color_palette("husl", n_colors=len(df_columns))
    
    # Plot each method
    for i, df_col in enumerate(df_columns):
        # Calculate mean and standard error for each unique variance explained value
        stats_by_var = df.groupby('var_explained').agg({
            df_col: ['mean', 'std', 'count']
        }).reset_index()
        
        # Sort by variance explained for proper line plotting
        stats_by_var = stats_by_var.sort_values('var_explained')
        
        # Calculate standard error
        stats_by_var['se'] = stats_by_var[(df_col, 'std')] / np.sqrt(stats_by_var[(df_col, 'count')])
        
        # Create label
        label = df_col.replace('df_', '').replace('_optimal', '').upper()
        
        # Plot mean line
        plt.plot(stats_by_var['var_explained'], 
                stats_by_var[(df_col, 'mean')], 
                marker='o',
                color=colors[i],
                label=label)
        
        # Add error bars (95% confidence interval)
        plt.fill_between(stats_by_var['var_explained'],
                        stats_by_var[(df_col, 'mean')] - 1.96 * stats_by_var['se'],
                        stats_by_var[(df_col, 'mean')] + 1.96 * stats_by_var['se'],
                        alpha=0.2,
                        color=colors[i])

    # Customize plot
    plt.xlabel('Proportion of Variance Explained', fontsize=12)
    plt.ylabel('Degrees of Freedom', fontsize=12)
    plt.title(f'Degrees of Freedom by Variance Explained (norm_beta={norm_beta})\n(Optimal configurations for gs and RGS)', 
              fontsize=14)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def extract_k_value(column_name):
    """Extract k value from column name"""
    return int(column_name.split('_k')[-1])

def extract_m_value(column_name):
    """Extract m value from column name"""
    return int(column_name.split('_m')[-1].split('_')[0])

def plot_df_vs_k(csv_path, target_sigma, tolerance=1e-6, save_path=None):
    """
    Plot df vs k curves for GS and RGS methods at a specific sigma value.
    
    Args:
        csv_path (str): Path to CSV file
        target_sigma (float): Target sigma value to plot
        tolerance (float): Tolerance for sigma value matching
        save_path (str, optional): Path to save plot
    """
    # Generate default save path if none provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/df_vs_k_sigma{target_sigma}_{timestamp}.png"
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows for target sigma
    mask = np.abs(df['sigma'] - target_sigma) < tolerance
    if not mask.any():
        raise ValueError(f"No data found for sigma = {target_sigma}")
    filtered_df = df[mask]
    
    # Create figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    # Plot GS
    gs_cols = [col for col in df.columns if col.startswith('df_gs_k')]
    k_values = [extract_k_value(col) for col in gs_cols]
    df_values = filtered_df[gs_cols].iloc[0]
    
    # Sort by k values
    k_df_pairs = sorted(zip(k_values, df_values))
    k_values_sorted = [k for k, _ in k_df_pairs]
    df_values_sorted = [d for _, d in k_df_pairs]
    
    # Plot GS in deep red
    ax.plot(k_values_sorted, df_values_sorted, 
            marker='o', 
            linewidth=2, 
            label='GS',
            color='#E63946',  # Deep Red for GS
            markersize=6)
    
    # Plot RGS for different m values, excluding m=250
    rgs_base_cols = [col for col in df.columns if col.startswith('df_rgs_m')]
    unique_m_values = sorted(list(set([extract_m_value(col) for col in rgs_base_cols])))
    unique_m_values = [m for m in unique_m_values if m != 250]  # Exclude m=250
    
    # Generate colors for RGS curves using a colormap
    # Using our color scheme but generating additional colors if needed
    base_colors = ['#1D3557', '#2A9D8F', '#F4A261', '#9B4F96']  # Our original colors
    if len(unique_m_values) <= len(base_colors):
        colors = base_colors[:len(unique_m_values)]
    else:
        # If we need more colors, generate them using a colormap
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_m_values)))
    
    for idx, m in enumerate(unique_m_values):
        rgs_cols_m = [col for col in rgs_base_cols if f'_m{m}_' in col]
        k_values = [extract_k_value(col) for col in rgs_cols_m]
        df_values = filtered_df[rgs_cols_m].iloc[0]
        
        # Sort by k values
        k_df_pairs = sorted(zip(k_values, df_values))
        k_values_sorted = [k for k, _ in k_df_pairs]
        df_values_sorted = [d for _, d in k_df_pairs]
        
        ax.plot(k_values_sorted, df_values_sorted, 
                marker='o', 
                linewidth=2, 
                label=f'RGS (m = {m})', 
                color=colors[idx],
                markersize=6)
    
    # Customize plot
    ax.set_xlabel('k', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degrees of Freedom', fontsize=12, fontweight='bold')
    ax.set_title(f'Degrees of Freedom vs k (σ = {target_sigma})', 
                fontsize=14, 
                fontweight='bold',
                pad=20)
    
    # Remove grid
    ax.grid(False)
    
    # Add legend in upper left
    ax.legend(loc='upper left', 
             frameon=True, 
             edgecolor='black',
             fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_mse_vs_k(csv_path, target_sigma, tolerance=1e-6, save_path=None):
    """
    Plot MSE vs k curves for gs and RGS methods at a specific sigma value.
    
    Args:
        csv_path (str): Path to CSV file
        target_sigma (float): Target sigma value to plot
        tolerance (float): Tolerance for sigma value matching
        save_path (str, optional): Path to save plot
    """
    # Generate default save path if none provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../figures/mse_vs_k_sigma{target_sigma}_{timestamp}.png"
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows for target sigma
    mask = np.abs(df['sigma'] - target_sigma) < tolerance
    if not mask.any():
        raise ValueError(f"No data found for sigma = {target_sigma}")
    filtered_df = df[mask]
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot gs (left subplot)
    gs_cols = [col for col in df.columns if col.startswith('mse_gs_k')]
    k_values = [extract_k_value(col) for col in gs_cols]
    
    mse_values = filtered_df[gs_cols].iloc[0]
    
    # Sort by k values
    k_mse_pairs = sorted(zip(k_values, mse_values))
    k_values_sorted = [k for k, _ in k_mse_pairs]
    mse_values_sorted = [m for _, m in k_mse_pairs]
    
    ax1.plot(k_values_sorted, mse_values_sorted, marker='o', linewidth=2, label='gs')
    
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Mean Square Error', fontsize=12)
    ax1.set_title(f'gs: MSE vs k (σ = {target_sigma})', fontsize=14)
    ax1.grid(True)
    
    # Plot baseline methods
    baseline_methods = ['mse_lasso', 'mse_ridge', 'mse_elastic']
    baseline_colors = ['r', 'g', 'b']
    baseline_labels = ['LASSO', 'Ridge', 'Elastic Net']
    
    for method, color, label in zip(baseline_methods, baseline_colors, baseline_labels):
        mse_value = filtered_df[method].iloc[0]
        ax1.axhline(y=mse_value, color=color, linestyle='--', label=label)
    
    ax1.legend()
    
    # Plot RGS (right subplot)
    rgs_base_cols = [col for col in df.columns if col.startswith('mse_rgs_m')]
    unique_m_values = sorted(list(set([extract_m_value(col) for col in rgs_base_cols])))
    
    # Create color palette for different m values
    colors = sns.color_palette("husl", n_colors=len(unique_m_values))
    
    for idx, m in enumerate(unique_m_values):
        rgs_cols_m = [col for col in rgs_base_cols if f'_m{m}_' in col]
        k_values = [extract_k_value(col) for col in rgs_cols_m]
        mse_values = filtered_df[rgs_cols_m].iloc[0]
        
        # Sort by k values
        k_mse_pairs = sorted(zip(k_values, mse_values))
        k_values_sorted = [k for k, _ in k_mse_pairs]
        mse_values_sorted = [m for _, m in k_mse_pairs]
        
        ax2.plot(k_values_sorted, mse_values_sorted, marker='o', 
                linewidth=2, label=f'm = {m}', color=colors[idx])
    
    # Add baseline methods to RGS plot
    for method, color, label in zip(baseline_methods, baseline_colors, baseline_labels):
        mse_value = filtered_df[method].iloc[0]
        ax2.axhline(y=mse_value, color=color, linestyle='--', label=label)
    
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Mean Square Error', fontsize=12)
    ax2.set_title(f'RGS: MSE vs k for different m (σ = {target_sigma})', fontsize=14)
    ax2.grid(True)
    ax2.legend(title='Window Size (m)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig