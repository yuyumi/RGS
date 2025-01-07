import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_k_value(col_name):
    """Safely extract k value from column name."""
    try:
        parts = col_name.split('_k')
        if len(parts) > 1:
            return int(parts[1].split('_')[0])
    except:
        return None

def get_m_value(col_name):
    """Safely extract m value from column name."""
    try:
        parts = col_name.split('_m')
        if len(parts) > 1:
            return int(parts[1].split('_')[0])
    except:
        return None

def find_best_methods(data, metric_prefix='pen'):
    """Find best performing FGS and RGS methods based on mean penalized."""
    # Find best FGS
    fgs_cols = [col for col in data.columns if col.startswith(f'{metric_prefix}_fgs_k')]
    best_fgs_col = min(fgs_cols, key=lambda x: data[x].mean())
    
    # Find best RGS
    rgs_cols = [col for col in data.columns if col.startswith(f'{metric_prefix}_rgs_m')]
    best_rgs_col = min(rgs_cols, key=lambda x: data[x].mean())
    
    return best_fgs_col, best_rgs_col

def save_fig(fig, filename, dpi=300):
    """Helper function to save figures with consistent settings."""
    fig.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def plot_mse_and_df_comparison(results_df, save=True):
    """
    Create side-by-side boxplots comparing MSE and degrees of freedom across methods and noise levels.
    """
    plt.style.use('seaborn')
    noise_levels = sorted(results_df['noise_level'].unique())
    
    # Create figure with two columns (MSE and DF) and rows for each noise level
    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(20, 6*len(noise_levels)))
    
    # Colors for different method types
    colors = {
        'baseline': 'skyblue',
        'fgs': 'lightgreen',
        'rgs': 'salmon'
    }
    
    for idx, noise in enumerate(noise_levels):
        noise_data = results_df[results_df['noise_level'] == noise]
        
        # Find best methods based on MSE
        best_fgs_col, best_rgs_col = find_best_methods(noise_data)
        
        # Plot MSE (left column)
        ax_mse = axes[idx, 0]
        plot_metric_boxplots(noise_data, 'mse', ax_mse, colors, best_fgs_col, best_rgs_col)
        ax_mse.set_title(f'MSE Distribution (σ = {noise.split("_")[1]})', fontsize=14)
        ax_mse.set_yscale('log')
        
        # Plot corresponding DF (right column)
        ax_df = axes[idx, 1]
        plot_metric_boxplots(noise_data, 'df', ax_df, colors, best_fgs_col, best_rgs_col)
        ax_df.set_title(f'Degrees of Freedom (σ = {noise.split("_")[1]})', fontsize=14)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=label) 
                      for label, color in colors.items()]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_fig(fig, f'../figures/boxplots_comparison_{timestamp}.png')
    
    return fig

def plot_mean_performance(results_df, save=True):
    """Plot mean MSE and DF for each method across different sigma values."""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get sigma values
    sigma_values = sorted(results_df['sigma'].unique())
    
    # Colors and markers for different method types
    styles = {
        'Lasso': ('skyblue', 'o'),
        'Ridge': ('skyblue', 's'),
        'Elastic': ('skyblue', '^'),
        'FGS': ('lightgreen', 'D'),
        'RGS': ('salmon', 'v')
    }
    
    # For each noise level, find best FGS and RGS methods
    mean_values_mse = {method: [] for method in ['Lasso', 'Ridge', 'Elastic', 'FGS', 'RGS']}
    mean_values_df = {method: [] for method in ['Lasso', 'Ridge', 'Elastic', 'FGS', 'RGS']}
    
    for sigma in sigma_values:
        sigma_data = results_df[results_df['sigma'] == sigma]
        
        # Baseline methods
        mean_values_mse['Lasso'].append(sigma_data['mse_lasso'].mean())
        mean_values_mse['Ridge'].append(sigma_data['mse_ridge'].mean())
        mean_values_mse['Elastic'].append(sigma_data['mse_elastic'].mean())
        
        mean_values_df['Lasso'].append(sigma_data['df_lasso'].mean())
        mean_values_df['Ridge'].append(sigma_data['df_ridge'].mean())
        mean_values_df['Elastic'].append(sigma_data['df_elastic'].mean())
        
        # Find best FGS method
        fgs_cols = [col for col in sigma_data.columns if col.startswith('pen_fgs_k')]
        best_fgs_col = min(fgs_cols, key=lambda x: sigma_data[x].mean())
        mean_values_mse['FGS'].append(sigma_data[best_fgs_col.replace('pen_', 'mse_')].mean())
        mean_values_df['FGS'].append(sigma_data[best_fgs_col.replace('pen_', 'df_')].mean())
        
        # Find best RGS method
        rgs_cols = [col for col in sigma_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_cols, key=lambda x: sigma_data[x].mean())
        mean_values_mse['RGS'].append(sigma_data[best_rgs_col.replace('pen_', 'mse_')].mean())
        mean_values_df['RGS'].append(sigma_data[best_rgs_col.replace('pen_', 'df_')].mean())
    
    # Plot MSE
    for method, (values_mse, values_df) in zip(mean_values_mse.keys(), 
                                             zip(mean_values_mse.values(), mean_values_df.values())):
        color, marker = styles[method]
        ax1.plot(sigma_values, values_mse, marker=marker, color=color, 
                label=method, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Mean MSE', fontsize=12)
    ax1.set_title('Mean MSE by Method', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot DF
    for method, (values_mse, values_df) in zip(mean_values_df.keys(), 
                                             zip(mean_values_mse.values(), mean_values_df.values())):
        color, marker = styles[method]
        ax2.plot(sigma_values, values_df, marker=marker, color=color, 
                label=method, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('Mean Degrees of Freedom', fontsize=12)
    ax2.set_title('Mean Degrees of Freedom by Method', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_fig(fig, f'../figures/mean_performance_{timestamp}.png')
    
    return fig

def create_all_plots(results_df):
    """Create and save all plots."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create and save boxplots
    fig1 = plot_mse_and_df_comparison(results_df, save=False)
    save_fig(fig1, f'../figures/boxplots_comparison_{timestamp}.png')
    
    # Create and save mean performance plots
    fig2 = plot_mean_performance(results_df, save=False)
    save_fig(fig2, f'../figures/mean_performance_{timestamp}.png')

def plot_k_performance_single(results_df, noise_level, save=True):
    """Plot MSE and DF against k_max for a single noise level."""
    plt.style.use('seaborn')
    
    # Create two separate figures for MSE and DF
    fig_mse = plt.figure(figsize=(12, 8))
    ax_mse = fig_mse.add_subplot(111)
    
    fig_df = plt.figure(figsize=(12, 8))
    ax_df = fig_df.add_subplot(111)
    
    noise_data = results_df[results_df['noise_level'] == noise_level]
    sigma = noise_level.split('_')[1]
    
    # Line styles for different methods
    fgs_style = {'color': 'black', 'linewidth': 2, 'marker': 'o', 'label': 'FGS'}
    
    # Get k values from FGS columns
    fgs_cols = [col for col in noise_data.columns if col.startswith('mse_fgs_k')]
    k_values = sorted(set(get_k_value(col) for col in fgs_cols))
    
    # Get m values from RGS columns
    rgs_cols = [col for col in noise_data.columns if col.startswith('mse_rgs_m')]
    m_values = sorted(set(get_m_value(col) for col in rgs_cols))
    rgs_colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    # Plot MSE
    mse_fgs = [noise_data[f'mse_fgs_k{k}'].mean() for k in k_values]
    ax_mse.plot(k_values, mse_fgs, **fgs_style)
    
    for idx, m in enumerate(m_values):
        mse_rgs = [noise_data[f'mse_rgs_m{m}_k{k}'].mean() for k in k_values]
        ax_mse.plot(k_values, mse_rgs, color=rgs_colors[idx], 
                   linewidth=2, marker='s', label=f'RGS m={m}')
    
    ax_mse.set_title(f'MSE vs k_max (σ = {sigma})', fontsize=14)
    ax_mse.set_xlabel('k_max', fontsize=12)
    ax_mse.set_ylabel('Mean MSE', fontsize=12)
    ax_mse.set_yscale('log')
    ax_mse.grid(True, linestyle='--', alpha=0.7)
    ax_mse.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Plot DF
    df_fgs = [noise_data[f'df_fgs_k{k}'].mean() for k in k_values]
    ax_df.plot(k_values, df_fgs, **fgs_style)
    
    for idx, m in enumerate(m_values):
        df_rgs = [noise_data[f'df_rgs_m{m}_k{k}'].mean() for k in k_values]
        ax_df.plot(k_values, df_rgs, color=rgs_colors[idx], 
                  linewidth=2, marker='s', label=f'RGS m={m}')
    
    ax_df.set_title(f'Degrees of Freedom vs k_max (σ = {sigma})', fontsize=14)
    ax_df.set_xlabel('k_max', fontsize=12)
    ax_df.set_ylabel('Mean Degrees of Freedom', fontsize=12)
    ax_df.grid(True, linestyle='--', alpha=0.7)
    ax_df.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Adjust layouts
    fig_mse.tight_layout(rect=[0, 0, 0.9, 1])
    fig_df.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_fig(fig_mse, f'../figures/mse_vs_k_{noise_level}_{timestamp}.png')
        save_fig(fig_df, f'../figures/df_vs_k_{noise_level}_{timestamp}.png')
    
    return fig_mse, fig_df

def plot_all_k_performance(results_df, save=True):
    """Create separate plots for each noise level."""
    noise_levels = sorted(results_df['noise_level'].unique())
    
    for noise in noise_levels:
        fig_mse, fig_df = plot_k_performance_single(results_df, noise, save=save)
        plt.close(fig_mse)
        plt.close(fig_df)

def plot_comparison_single(data, noise_level, save=True):
    """Create separate boxplots for MSE and DF for a single noise level."""
    plt.style.use('seaborn')
    sigma = noise_level.split('_')[1]
    
    # Colors for different method types
    colors = {
        'baseline': 'skyblue',
        'fgs': 'lightgreen',
        'rgs': 'salmon'
    }
    
    # Create MSE figure
    fig_mse = plt.figure(figsize=(12, 8))
    ax_mse = fig_mse.add_subplot(111)
    best_fgs_col, best_rgs_col = find_best_methods(data)
    plot_metric_boxplots(data, 'mse', ax_mse, colors, best_fgs_col, best_rgs_col)
    ax_mse.set_title(f'MSE Distribution (σ = {sigma})', fontsize=14)
    ax_mse.set_yscale('log')
    
    # Create DF figure
    fig_df = plt.figure(figsize=(12, 8))
    ax_df = fig_df.add_subplot(111)
    plot_metric_boxplots(data, 'df', ax_df, colors, best_fgs_col, best_rgs_col)
    ax_df.set_title(f'Degrees of Freedom (σ = {sigma})', fontsize=14)
    
    # Add legends
    for fig in [fig_mse, fig_df]:
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=label) 
                         for label, color in colors.items()]
        fig.legend(handles=legend_elements, loc='center right', 
                  bbox_to_anchor=(0.98, 0.5))
        fig.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_fig(fig_mse, f'../figures/mse_boxplot_{noise_level}_{timestamp}.png')
        save_fig(fig_df, f'../figures/df_boxplot_{noise_level}_{timestamp}.png')
    
    return fig_mse, fig_df

def plot_all_comparisons(results_df, save=True):
    """Create separate boxplots for each noise level."""
    noise_levels = sorted(results_df['noise_level'].unique())
    
    for noise in noise_levels:
        noise_data = results_df[results_df['noise_level'] == noise]
        fig_mse, fig_df = plot_comparison_single(noise_data, noise, save=save)
        plt.close(fig_mse)
        plt.close(fig_df)

def plot_mean_performance_single(results_df, save=True):
    """Create separate MSE and DF plots."""
    plt.style.use('seaborn')
    
    # Get sigma values
    sigma_values = sorted(results_df['sigma'].unique())
    
    # Colors and markers for different method types
    styles = {
        'Lasso': ('skyblue', 'o'),
        'Ridge': ('skyblue', 's'),
        'Elastic': ('skyblue', '^'),
        'FGS': ('lightgreen', 'D'),
        'RGS': ('salmon', 'v')
    }
    
    # Calculate mean values
    mean_values_mse = {method: [] for method in ['Lasso', 'Ridge', 'Elastic', 'FGS', 'RGS']}
    mean_values_df = {method: [] for method in ['Lasso', 'Ridge', 'Elastic', 'FGS', 'RGS']}
    
    for sigma in sigma_values:
        sigma_data = results_df[results_df['sigma'] == sigma]
        
        # Baseline methods
        mean_values_mse['Lasso'].append(sigma_data['mse_lasso'].mean())
        mean_values_mse['Ridge'].append(sigma_data['mse_ridge'].mean())
        mean_values_mse['Elastic'].append(sigma_data['mse_elastic'].mean())
        
        mean_values_df['Lasso'].append(sigma_data['df_lasso'].mean())
        mean_values_df['Ridge'].append(sigma_data['df_ridge'].mean())
        mean_values_df['Elastic'].append(sigma_data['df_elastic'].mean())
        
        # Find best FGS method
        fgs_cols = [col for col in sigma_data.columns if col.startswith('pen_fgs_k')]
        best_fgs_col = min(fgs_cols, key=lambda x: sigma_data[x].mean())
        mean_values_mse['FGS'].append(sigma_data[best_fgs_col.replace('pen_', 'mse_')].mean())
        mean_values_df['FGS'].append(sigma_data[best_fgs_col.replace('pen_', 'df_')].mean())
        
        # Find best RGS method
        rgs_cols = [col for col in sigma_data.columns if col.startswith('pen_rgs_m')]
        best_rgs_col = min(rgs_cols, key=lambda x: sigma_data[x].mean())
        mean_values_mse['RGS'].append(sigma_data[best_rgs_col.replace('pen_', 'mse_')].mean())
        mean_values_df['RGS'].append(sigma_data[best_rgs_col.replace('pen_', 'df_')].mean())
    
    # Create MSE plot
    fig_mse = plt.figure(figsize=(12, 8))
    ax_mse = fig_mse.add_subplot(111)
    
    for method, values in mean_values_mse.items():
        color, marker = styles[method]
        ax_mse.plot(sigma_values, values, marker=marker, color=color, 
                   label=method, linewidth=2, markersize=8)
    
    ax_mse.set_xlabel('Noise Level (σ)', fontsize=12)
    ax_mse.set_ylabel('Mean MSE', fontsize=12)
    ax_mse.set_title('Mean MSE by Method', fontsize=14)
    ax_mse.set_yscale('log')
    ax_mse.grid(True, linestyle='--', alpha=0.7)
    ax_mse.legend(fontsize=10)
    
    # Create DF plot
    fig_df = plt.figure(figsize=(12, 8))
    ax_df = fig_df.add_subplot(111)
    
    for method, values in mean_values_df.items():
        color, marker = styles[method]
        ax_df.plot(sigma_values, values, marker=marker, color=color, 
                  label=method, linewidth=2, markersize=8)
    
    ax_df.set_xlabel('Noise Level (σ)', fontsize=12)
    ax_df.set_ylabel('Mean Degrees of Freedom', fontsize=12)
    ax_df.set_title('Mean Degrees of Freedom by Method', fontsize=14)
    ax_df.grid(True, linestyle='--', alpha=0.7)
    ax_df.legend(fontsize=10)
    
    # Adjust layouts
    fig_mse.tight_layout()
    fig_df.tight_layout()
    
    if save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_fig(fig_mse, f'../figures/mean_mse_vs_sigma_{timestamp}.png')
        save_fig(fig_df, f'../figures/mean_df_vs_sigma_{timestamp}.png')
    
    return fig_mse, fig_df