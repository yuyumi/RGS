import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Optional, List, Dict
from pathlib import Path

__all__ = [
    'plot_mse_by_sigma',
    'plot_df_by_sigma',
    'plot_insample_by_sigma',
    'plot_outsample_mse_by_sigma',
    'plot_mse_by_variance_explained',
    'plot_df_by_variance_explained',
    'plot_insample_by_variance_explained',
    'plot_outsample_mse_by_variance_explained',
    'plot_mse_vs_k',
    'plot_df_vs_k',
    'plot_insample_vs_k',
    'plot_outsample_mse_vs_k',
    # New bar plot functions
    'barplot_metric_by_sigma',
    'barplot_metric_by_variance_explained',
    'barplot_metric_vs_k',
    'barplot_mse_by_sigma',
    'barplot_df_by_sigma',
    'barplot_insample_by_sigma',
    'barplot_outsample_mse_by_sigma',
    'barplot_mse_by_variance_explained',
    'barplot_df_by_variance_explained',
    'barplot_insample_by_variance_explained',
    'barplot_outsample_mse_by_variance_explained',
    'barplot_mse_vs_k',
    'barplot_df_vs_k',
    'barplot_insample_vs_k',
    'barplot_outsample_mse_vs_k',
    'plot_mse_vs_df',
    'plot_df_by_k'
]

class PlottingConfig:
    """Configuration class for plotting parameters."""
    COLORS = {
        # Gradient of blues for Lasso, Ridge, Elastic
        'lasso': '#1A5276',  # Deep blue
        'ridge': '#2980B9',  # Medium blue
        'elastic': '#7FB3D5',  # Light blue
        
        # Gradient of oranges/reds for GS methods (colorblind-friendly)
        'bagged_gs': '#922B21',  # Deep red
        'smeared_gs': '#CB4335',  # Medium red
        'base_gs': '#E74C3C',     # Regular red
        'original_gs': '#F39C12',  # Orange
        'rgs': '#F8C471'          # Light orange
    }
    
    METRIC_LABELS = {
        'mse': 'Mean Square Error',
        'insample': 'In-sample Error',
        'outsample_mse': 'Out-of-sample Mean Square Error',
        'df': 'Degrees of Freedom',
        'coef_recovery': 'Coefficient Recovery Error',
        'support_recovery': 'Support Recovery Accuracy'
    }
    
    # Methods: Lasso, Ridge, Elastic Net, Greedy Selection, Randomized Greedy Selection
    METHODS = ['lasso', 
               'ridge',
                'elastic', 
                'bagged_gs', 'smeared_gs',
                # 'base_rgs', 'base_gs',
                'original_gs', 'rgs']
    
    METHOD_LABELS = {
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'elastic': 'Elastic',
        'bagged_gs': 'Bagged GS',
        'smeared_gs': 'Smeared GS',
        'base_rgs': 'Base RGS',
        'base_gs': 'Base GS',
        'original_gs': 'GS',  # Changed from 'Original_GS' to just 'GS'
        'rgs': 'RGS'
    }
    
    @classmethod
    def get_metric_label(cls, metric: str) -> str:
        """Get formatted label for a given metric."""
        return cls.METRIC_LABELS.get(metric, metric.replace('_', ' ').title())
    
    @classmethod
    def get_method_label(cls, method: str) -> str:
        """Get formatted label for a method."""
        return cls.METHOD_LABELS.get(method, method.upper())

def get_available_methods(df: pd.DataFrame, metric_prefix: str) -> List[str]:
    """Get list of available methods based on column prefixes."""
    return [m for m in PlottingConfig.METHODS if f'{metric_prefix}_{m}' in df.columns]

def setup_plot_style() -> None:
    """Set up consistent plot styling."""
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })

def create_figure() -> tuple:
    """Create and return a new figure with standard size."""
    return plt.subplots(figsize=(7, 5))

def enhance_log_axis(ax):
    """Apply consistent log axis formatting with specific labeled ticks."""
    # Set major ticks at powers of 10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    
    # Create specific tick values for the y-axis that look good on a log scale
    def specific_log_ticks(vmin, vmax):
        ticks = []
        base = 10.0
        # Find the decade range
        if vmin <= 0:
            vmin = 1e-10  # Small positive value
        
        # Determine the exponent range
        vmin_log = np.log(vmin) / np.log(base)
        vmax_log = np.log(vmax) / np.log(base)
        
        start_exp = np.floor(vmin_log)
        end_exp = np.ceil(vmax_log)
        
        # Only use clean decimal values like 0.1, 0.2, 0.5, 1.0, 2.0, 5.0
        for exp in range(int(start_exp), int(end_exp) + 1):
            # Main decade marker
            decade = base ** exp
            ticks.append(decade)
            
            # Add nice intermediate values
            if exp < end_exp:  # Don't add intermediates after the last full decade
                for multiplier in [2, 5]:
                    tick = multiplier * decade
                    if vmin <= tick <= vmax:
                        ticks.append(tick)
        
        return np.array(sorted(ticks))
    
    # Create a manual locator that uses our function to generate tick positions
    class ManualLogLocator(ticker.Locator):
        def __init__(self, tick_function):
            self.tick_function = tick_function
            
        def tick_values(self, vmin, vmax):
            return self.tick_function(vmin, vmax)
            
        def __call__(self):
            # Get the current axis limits
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)
    
    # Apply our custom locator
    ax.yaxis.set_major_locator(ManualLogLocator(specific_log_ticks))
    
    # Use a formatter that shows decimal values clearly
    def log_format(x, pos):
        if x >= 1:
            return f"{x:.1f}"
        elif x >= 0.1:
            return f"{x:.1f}"
        else:
            return f"{x:.2f}"
            
    formatter = ticker.FuncFormatter(log_format)
    ax.yaxis.set_major_formatter(formatter)
    
    # Add minor ticks for better readability
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    
    # Ensure grid lines
    ax.grid(which='major', linestyle='-', alpha=0.3)
    ax.grid(which='minor', linestyle=':', alpha=0.1)

def format_x_axis_decimal(ax):
    """Format x-axis to show values at 0.1 intervals."""
    # Set x-axis ticks at 0.1 intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # Format with one decimal place
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # Add minor ticks at 0.05 intervals
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # Add grid
    ax.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', alpha=0.2)

def plot_metric_by_sigma(
    results_path: Path,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    show_std: bool = False
) -> Optional[plt.Figure]:
    """Plot metric vs sigma with error bands."""
    try:
        df = pd.read_csv(results_path)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods:
            print(f"No data found for metric '{metric}' in {Path(results_path).name}")
            return None
            
        # Check if there's any valid data for the available methods
        valid_data = False
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            if not df[metric_col].isna().all():
                valid_data = True
                break
                
        if not valid_data:
            print(f"All data for metric '{metric}' is NaN in {Path(results_path).name}")
            return None
            
        fig, ax = create_figure()
        setup_plot_style()
        
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            grouped = df.groupby('sigma')[metric_col].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'],
                marker='o',
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method))
                   
            if show_std:
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        enhance_log_axis(ax)
        
        ax.set_xlabel('Sigma (Noise Level)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        # Title removed
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by sigma: {str(e)}")
        plt.close()
        return None

def plot_metric_by_variance_explained(
    results_path: Path,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    show_std: bool = True,
    norm_beta: float = 10.0
) -> Optional[plt.Figure]:
    """Plot metric vs proportion of variance explained with error bands."""
    try:
        df = pd.read_csv(results_path)
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
        
        fig, ax = create_figure()
        setup_plot_style()
        
        available_methods = get_available_methods(df, metric)
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
            
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            grouped = df.groupby('var_explained')[metric_col].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'],
                marker='o',
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method))
                   
            if show_std:
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        enhance_log_axis(ax)
        
        # Format x-axis with decimal intervals
        format_x_axis_decimal(ax)
        
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        # Title removed
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse'] else 'upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by variance explained: {str(e)}")
        plt.close()
        return None

def plot_metric_vs_k(
    results_path: Path,
    target_sigma: float,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    sigma_tolerance: float = 1e-6
) -> Optional[plt.Figure]:
    """Plot metric vs k for a specific sigma value."""
    try:
        df = pd.read_csv(results_path)
        df_sigma = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
        
        if df_sigma.empty:
            print(f"No data found for σ = {target_sigma} in {Path(results_path).name}")
            return None
            
        available_methods = get_available_methods(df_sigma, metric)
        if not available_methods:
            print(f"No data found for metric '{metric}' in {Path(results_path).name}")
            return None
            
        # Check if there's any valid data for the available methods
        valid_data = False
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            if not df_sigma[metric_col].isna().all():
                valid_data = True
                break
                
        if not valid_data:
            print(f"All data for metric '{metric}' is NaN for σ = {target_sigma} in {Path(results_path).name}")
            return None
        
        fig, ax = create_figure()
        setup_plot_style()
        
        baseline_methods = [m for m in ['lasso', 'ridge', 'elastic'] if m in available_methods]
        advanced_methods = [m for m in ['gs', 'rgs'] if m in available_methods]
        
        if not baseline_methods and not advanced_methods:
            print(f"No valid methods found for σ = {target_sigma} in {Path(results_path).name}")
            plt.close(fig)
            return None
        
        # Plot baseline methods
        for method in baseline_methods:
            metric_col = f'{metric}_{method}'
            metric_value = df_sigma[metric_col].mean()
            ax.axhline(y=metric_value, 
                color=PlottingConfig.COLORS[method],
                linestyle='--',
                label=PlottingConfig.get_method_label(method))
        
        # Plot k-dependent methods
        if 'best_k' in df_sigma.columns:
            k_values = sorted(df_sigma['best_k'].unique())
            
            for method in advanced_methods:
                metric_col = f'{metric}_{method}'
                k_metrics = []
                k_stds = []
                
                for k in k_values:
                    k_data = df_sigma[df_sigma['best_k'] == k][metric_col]
                    k_metrics.append(k_data.mean())
                    k_stds.append(k_data.std())
                
                ax.plot(k_values, k_metrics,
                        marker='o',
                        color=PlottingConfig.COLORS[method],
                        label=PlottingConfig.get_method_label(method))
                       
                ax.fill_between(
                    k_values,
                    np.array(k_metrics) - np.array(k_stds),
                    np.array(k_metrics) + np.array(k_stds),
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        enhance_log_axis(ax)
        
        ax.set_xlabel('k')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        # Modified to include sigma in the axis label instead of title
        ax.set_xlabel(f'k (σ = {target_sigma:.2f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting metric vs k: {str(e)}")
        plt.close()
        return None
    
def plot_mse_vs_df(
    results_path: Path, 
    target_sigma: Optional[float] = None,
    save_path: Optional[Path] = None,
    show_std: bool = True,
    log_scale_mse: bool = True,
    log_scale_df: bool = False,
    sigma_tolerance: float = 1e-6,
    plot_all_sigmas: bool = True,  # New parameter to force plotting all sigmas on one plot
    method_filter: Optional[callable] = None  # Filter to exclude certain methods
) -> Optional[plt.Figure]:
    """
    Plot MSE vs DF for each method, with all sigma values on the same plot.
    
    Parameters:
    -----------
    results_path : Path
        Path to the CSV file containing simulation results
    target_sigma : float, optional
        If provided and plot_all_sigmas is False, plot only results for this specific sigma value
    save_path : Path, optional
        Path to save the figure
    show_std : bool
        Whether to show standard deviation as error bars
    log_scale_mse : bool
        Whether to use log scale for MSE axis
    log_scale_df : bool
        Whether to use log scale for DF axis
    sigma_tolerance : float
        Tolerance for identifying the target sigma in the data
    plot_all_sigmas : bool
        If True, plot all sigma values on the same plot (default behavior)
        
    Returns:
    --------
    matplotlib.Figure or None
        The generated figure or None if an error occurred
    """
    try:
        from rgs_experiments.plotting.plotting import (
            PlottingConfig, 
            get_available_methods, 
            setup_plot_style,
            create_figure, 
            enhance_log_axis
        )
        
        df = pd.read_csv(results_path)
        
        # Filter by target sigma if provided and not plotting all sigmas
        if target_sigma is not None and not plot_all_sigmas:
            df = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
            if df.empty:
                print(f"No data found for σ = {target_sigma} in {Path(results_path).name}")
                return None
        
        # Check for required columns
        for metric in ['mse', 'df']:
            available_methods = get_available_methods(df, metric)
            if not available_methods:
                print(f"No data found for metric '{metric}' in {Path(results_path).name}")
                return None
        
        # Get methods that have both MSE and DF data
        mse_methods = get_available_methods(df, 'mse')
        df_methods = get_available_methods(df, 'df')
        methods = [m for m in mse_methods if m in df_methods]
        
        # Apply method filter if provided
        if method_filter:
            methods = [m for m in methods if method_filter(m)]
        
        if not methods:
            print(f"No methods with both MSE and DF data in {Path(results_path).name}")
            return None
        
        # Create figure
        fig, ax = create_figure()
        setup_plot_style()
        
        # Get all sigma values
        sigma_values = sorted(df['sigma'].unique())
        
        # Create curves for each method connecting points across sigma values
        for method in methods:
            mse_col = f'mse_{method}'
            df_col = f'df_{method}'
            
            # Extract data points for this method (one per sigma)
            points = []
            for sigma in sigma_values:
                sigma_data = df[df['sigma'] == sigma]
                
                # Always use MSE for training loss
                error_mean = sigma_data[mse_col].mean()
                error_std = sigma_data[mse_col].std() if show_std else 0
                
                df_mean = sigma_data[df_col].mean()
                df_std = sigma_data[df_col].std() if show_std else 0
                
                points.append((sigma, error_mean, df_mean, error_std, df_std))
            
            # Plot the curve connecting all sigma points for this method
            sigmas, error_means, df_means, error_stds, df_stds = zip(*points)
            
            # Create a line connecting the points
            ax.plot(
                error_means, 
                df_means,
                '-',  # Just use a line
                color=PlottingConfig.COLORS[method],
                alpha=0.5,
                linewidth=1.5,
                label=PlottingConfig.get_method_label(method)
            )
            
            # Plot each point with a marker
            for i, sigma in enumerate(sigmas):
                # Use a different marker shape for each method
                marker_style = 'o' if 'gs' in method.lower() or method == 'rgs' else 'X'
                
                # Plot point
                ax.scatter(
                    error_means[i], 
                    df_means[i],
                    color=PlottingConfig.COLORS[method],
                    marker=marker_style,
                    s=80,
                    zorder=3,
                    edgecolors='black',
                    linewidths=0.5
                )
        
        # Set scales
        if log_scale_mse:
            ax.set_xscale('log')
            # Apply custom formatting to x-axis
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())
        
        if log_scale_df:
            ax.set_yscale('log')
            # Apply custom formatting to y-axis
            from matplotlib.ticker import ScalarFormatter
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Set labels
        ax.set_xlabel('Mean Square Error (MSE) - Training Loss')
        ax.set_ylabel('Degrees of Freedom (DF)')
        
        # Set title
        ax.set_title('Model Complexity vs. Training Loss')
        
        # Create custom legend with method information
        handles, labels = ax.get_legend_handles_labels()
        gs_methods = [i for i, m in enumerate(methods) if 'gs' in m.lower() or m == 'rgs']
        reg_methods = [i for i, m in enumerate(methods) if i not in gs_methods]
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Save or return figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig
    
    except Exception as e:
        print(f"Error plotting MSE vs DF: {str(e)}")
        plt.close()
        return None

# New bar plotting functions
def barplot_metric_by_sigma(
    results_path: Path,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    show_std: bool = True,
    log_scale: bool = True
) -> Optional[plt.Figure]:
    """Create bar plot of metric vs sigma with error bars."""
    try:
        df = pd.read_csv(results_path)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods:
            print(f"No data found for metric '{metric}' in {Path(results_path).name}")
            return None
            
        # Check if there's any valid data for the available methods
        valid_data = False
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            if not df[metric_col].isna().all():
                valid_data = True
                break
                
        if not valid_data:
            print(f"All data for metric '{metric}' is NaN in {Path(results_path).name}")
            return None
            
        fig, ax = create_figure()
        setup_plot_style()
        
        sigma_values = sorted(df['sigma'].unique())
        n_methods = len(available_methods)
        n_sigmas = len(sigma_values)
        
        # Calculate bar positions
        bar_width = 0.8 / n_methods
        
        # First, group methods by type (GS-based vs non-GS)
        gs_methods = [m for m in available_methods if 'gs' in m.lower() or m.lower() == 'rgs']
        non_gs_methods = [m for m in available_methods if m not in gs_methods]
        
        # Sort each group to maintain color progression
        gs_methods.sort()
        non_gs_methods.sort()
        
        # Combine sorted groups
        sorted_methods = non_gs_methods + gs_methods
        
        for i, method in enumerate(sorted_methods):
            metric_col = f'{metric}_{method}'
            
            means = []
            stds = []
            positions = []
            
            for j, sigma in enumerate(sigma_values):
                sigma_data = df[df['sigma'] == sigma][metric_col]
                means.append(sigma_data.mean())
                stds.append(sigma_data.std())
                # Position bars for each method group
                positions.append(j + (i - n_methods/2 + 0.5) * bar_width)
            
            # Create bars
            bars = ax.bar(
                positions, 
                means,
                width=bar_width,
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method),
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add error bars if requested
            if show_std:
                ax.errorbar(
                    positions,
                    means,
                    yerr=stds,
                    fmt='none',
                    color='black',
                    capsize=3,
                    linewidth=1,
                    capthick=1
                )
        
        # Set y-scale to log if requested
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
        else:
            # For linear scale, ensure y-axis starts from 0
            ax.set_ylim(bottom=0)
            
            # Add proper y-tick formatting for linear scale
            formatter = ticker.FormatStrFormatter('%.2f')
            ax.yaxis.set_major_formatter(formatter)
            
            # Set major ticks at reasonable intervals
            if metric in ['mse', 'insample', 'outsample_mse']:
                max_val = max([max(df[f'{metric}_{m}']) for m in available_methods if f'{metric}_{m}' in df.columns])
                tick_interval = max(0.1, max_val / 10)  # Reasonable interval based on data range
                ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            
            # Add minor ticks for better gridlines
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            
            # Add grid
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        # Set the x-ticks to be centered for each sigma group
        ax.set_xticks(range(n_sigmas))
        ax.set_xticklabels([f"{sigma:.2f}" for sigma in sigma_values])
        
        ax.set_xlabel('Sigma (Noise Level)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by sigma: {str(e)}")
        plt.close()
        return None

def barplot_metric_by_variance_explained(
    results_path: Path,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    show_std: bool = True,
    log_scale: bool = True,
    norm_beta: float = 10.0
) -> Optional[plt.Figure]:
    """Create bar plot of metric vs proportion of variance explained with error bars."""
    try:
        df = pd.read_csv(results_path)
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
        
        fig, ax = create_figure()
        setup_plot_style()
        
        available_methods = get_available_methods(df, metric)
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
            
        # Round var_explained to 2 decimal places for grouping
        df['var_explained_rounded'] = np.round(df['var_explained'], 2)
        var_explained_values = sorted(df['var_explained_rounded'].unique())
        
        n_methods = len(available_methods)
        n_vars = len(var_explained_values)
        
        # Calculate bar positions
        bar_width = 0.8 / n_methods
        
        for i, method in enumerate(available_methods):
            metric_col = f'{metric}_{method}'
            
            means = []
            stds = []
            positions = []
            
            for j, var_expl in enumerate(var_explained_values):
                var_data = df[np.isclose(df['var_explained_rounded'], var_expl)][metric_col]
                means.append(var_data.mean())
                stds.append(var_data.std())
                # Position bars for each method group
                positions.append(j + (i - n_methods/2 + 0.5) * bar_width)
            
            # Create bars
            bars = ax.bar(
                positions, 
                means,
                width=bar_width,
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method),
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add error bars if requested
            if show_std:
                ax.errorbar(
                    positions,
                    means,
                    yerr=stds,
                    fmt='none',
                    color='black',
                    capsize=3,
                    linewidth=1,
                    capthick=1
                )
        
        # Set y-scale to log if requested
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
        else:
            # For linear scale, ensure y-axis starts from 0
            ax.set_ylim(bottom=0)
            
            # Add proper y-tick formatting for linear scale
            formatter = ticker.FormatStrFormatter('%.2f')
            ax.yaxis.set_major_formatter(formatter)
            
            # Set major ticks at reasonable intervals
            if metric in ['mse', 'insample', 'outsample_mse']:
                max_val = max([max(df[f'{metric}_{m}']) for m in available_methods if f'{metric}_{m}' in df.columns])
                tick_interval = max(0.1, max_val / 10)  # Reasonable interval based on data range
                ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
            
            # Add minor ticks for better gridlines
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            
            # Add grid
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        # Set the x-ticks to be centered for each var_explained group
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels([f"{var:.2f}" for var in var_explained_values])
        
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse'] else 'upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by variance explained: {str(e)}")
        plt.close()
        return None

def barplot_metric_vs_k(
    results_path: Path,
    target_sigma: float,
    metric: str = 'mse',
    save_path: Optional[Path] = None,
    show_std: bool = True,
    log_scale: bool = True,
    sigma_tolerance: float = 1e-6
) -> Optional[plt.Figure]:
    """Create bar plot of metric vs k for a specific sigma value."""
    try:
        df = pd.read_csv(results_path)
        df_sigma = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
        
        if df_sigma.empty:
            print(f"No data found for σ = {target_sigma} in {Path(results_path).name}")
            return None
            
        available_methods = get_available_methods(df_sigma, metric)
        if not available_methods:
            print(f"No data found for metric '{metric}' in {Path(results_path).name}")
            return None
            
        # Check if there's any valid data for the available methods
        valid_data = False
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            if not df_sigma[metric_col].isna().all():
                valid_data = True
                break
                
        if not valid_data:
            print(f"All data for metric '{metric}' is NaN for σ = {target_sigma} in {Path(results_path).name}")
            return None
        
        fig, ax = create_figure()
        setup_plot_style()
        
        # Separate baseline methods and k-dependent methods
        baseline_methods = [m for m in ['lasso', 'ridge', 'elastic'] if m in available_methods]
        advanced_methods = [m for m in available_methods if m not in baseline_methods]
        
        # Extract k values if they exist
        if 'best_k' in df_sigma.columns:
            k_values = sorted(df_sigma['best_k'].unique())
            n_methods = len(available_methods)
            
            # Calculate positions for the bars
            # First position: baseline methods
            # Remaining positions: k-dependent methods by k
            
            # Calculate how many positions we'll need in total
            # For baseline methods, we use one position per method
            # For k-dependent methods, we use n_k positions for each method
            total_baseline_pos = len(baseline_methods)
            total_k_pos = len(k_values) * len(advanced_methods)
            total_pos = total_baseline_pos + total_k_pos
            
            # Start with baseline methods
            current_pos = 0
            bar_width = 0.8
            
            # Plot baseline methods as single bars
            for method in baseline_methods:
                metric_col = f'{metric}_{method}'
                method_data = df_sigma[metric_col]
                
                # Create a bar for this method
                ax.bar(
                    current_pos,
                    method_data.mean(),
                    width=bar_width,
                    color=PlottingConfig.COLORS[method],
                    label=PlottingConfig.get_method_label(method),
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Add error bar
                if show_std:
                    ax.errorbar(
                        current_pos,
                        method_data.mean(),
                        yerr=method_data.std(),
                        fmt='none',
                        color='black',
                        capsize=3,
                        linewidth=1,
                        capthick=1
                    )
                    
                current_pos += 1
            
            # Add a small gap between baseline methods and k-dependent methods
            if baseline_methods and advanced_methods:
                current_pos += 0.5
            
            # Now plot k-dependent methods, grouped by method
            for method in advanced_methods:
                metric_col = f'{metric}_{method}'
                
                for k in k_values:
                    k_data = df_sigma[df_sigma['best_k'] == k][metric_col]
                    
                    # Create a bar for this method at this k
                    ax.bar(
                        current_pos,
                        k_data.mean(),
                        width=bar_width,
                        color=PlottingConfig.COLORS[method],
                        label=f"{PlottingConfig.get_method_label(method)} (k={k})" if k == k_values[0] else "",
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Add a text label for k
                    ax.text(current_pos, 0, f"k={k}", ha='center', va='bottom', rotation=90)
                    
                    # Add error bar
                    if show_std:
                        ax.errorbar(
                            current_pos,
                            k_data.mean(),
                            yerr=k_data.std(),
                            fmt='none',
                            color='black',
                            capsize=3,
                            linewidth=1,
                            capthick=1
                        )
                    
                    current_pos += 1
                
                # Add a small gap between different methods
                if method != advanced_methods[-1]:
                    current_pos += 0.5
        else:
            # If there are no k values, just plot all methods as regular bars
            n_methods = len(available_methods)
            bar_width = 0.8
            
            for i, method in enumerate(available_methods):
                metric_col = f'{metric}_{method}'
                method_data = df_sigma[metric_col]
                
                ax.bar(
                    i,
                    method_data.mean(),
                    width=bar_width,
                    color=PlottingConfig.COLORS[method],
                    label=PlottingConfig.get_method_label(method),
                    edgecolor='black',
                    linewidth=0.5
                )
                
                if show_std:
                    ax.errorbar(
                        i,
                        method_data.mean(),
                        yerr=method_data.std(),
                        fmt='none',
                        color='black',
                        capsize=3,
                        linewidth=1,
                        capthick=1
                    )
        
        # Set y-scale to log if requested
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
        
        # Remove x ticks
        ax.set_xticks([])
        
        ax.set_xlabel(f'Method (σ = {target_sigma:.2f})')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot vs k: {str(e)}")
        plt.close()
        return None
    
def plot_df_by_k(
    results_path: Path,
    target_sigma: float,
    save_path: Optional[Path] = None,
    show_std: bool = True,
    sigma_tolerance: float = 1e-6
) -> Optional[plt.Figure]:
    """Plot degrees of freedom vs k for different models at a specific sigma value."""
    try:
        df = pd.read_csv(results_path)
        df_sigma = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
        
        if df_sigma.empty:
            print(f"No data found for σ = {target_sigma} in {Path(results_path).name}")
            return None
            
        # Check if there are any df_by_k columns
        df_by_k_columns = [col for col in df_sigma.columns if col.startswith('df_by_k_')]
        if not df_by_k_columns:
            print(f"No df_by_k data found in {Path(results_path).name}")
            return None
        
        fig, ax = create_figure()
        setup_plot_style()
        
        # Get list of methods
        methods = set()
        for col in df_by_k_columns:
            # Extract method from column name (format: df_by_k_method_k)
            parts = col.split('_')
            if len(parts) >= 4:
                method = '_'.join(parts[2:-1])  # Handle methods with underscores
                methods.add(method)
        
        # Filter to only use methods that are in PlottingConfig.METHODS
        available_methods = [m for m in methods if m in PlottingConfig.METHODS]
        
        for method in available_methods:
            # Get all columns for this method
            method_columns = [col for col in df_by_k_columns if col.startswith(f'df_by_k_{method}_')]
            
            # Extract k values from column names
            k_values = []
            for col in method_columns:
                try:
                    k = int(col.split('_')[-1])
                    k_values.append(k)
                except ValueError:
                    continue
            
            if not k_values:
                continue
                
            # Sort k values
            k_values = sorted(k_values)
            
            # Calculate mean and std of df for each k
            means = []
            stds = []
            
            for k in k_values:
                col = f'df_by_k_{method}_{k}'
                if col in df_sigma.columns:
                    means.append(df_sigma[col].mean())
                    stds.append(df_sigma[col].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            # Plot the curve
            ax.plot(
                k_values, 
                means,
                marker='o',
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method)
            )
            
            if show_std:
                ax.fill_between(
                    k_values,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        # Add diagonal line showing k = df (theoretical OLS line)
        max_k = max(k_values) if k_values else 0
        ax.plot([0, max_k], [0, max_k], 'k--', alpha=0.7, label='k (OLS)')
        
        ax.set_xlabel(f'k (Number of Features, σ = {target_sigma:.2f})')
        ax.set_ylabel('Effective Degrees of Freedom')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig
        
    except Exception as e:
        print(f"Error plotting df by k: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close()
        return None

# Convenience functions for original line plots
def plot_mse_by_sigma(*args, **kwargs):
    return plot_metric_by_sigma(*args, metric='mse', **kwargs)

def plot_df_by_sigma(*args, **kwargs):
    return plot_metric_by_sigma(*args, metric='df', **kwargs)

def plot_insample_by_sigma(*args, **kwargs):
    return plot_metric_by_sigma(*args, metric='insample', **kwargs)

def plot_outsample_mse_by_sigma(*args, **kwargs):
    return plot_metric_by_sigma(*args, metric='outsample_mse', **kwargs)

def plot_mse_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def plot_df_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='df', **kwargs)

def plot_insample_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def plot_outsample_mse_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='outsample_mse', **kwargs)

def plot_mse_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='mse', **kwargs)

def plot_df_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='df', **kwargs)

def plot_insample_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='insample', **kwargs)

def plot_outsample_mse_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='outsample_mse', **kwargs)

# Convenience functions for bar plots
def barplot_mse_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='mse', **kwargs)

def barplot_df_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='df', **kwargs)

def barplot_insample_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='insample', **kwargs)

def barplot_outsample_mse_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='outsample_mse', **kwargs)

def barplot_mse_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def barplot_df_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='df', **kwargs)

def barplot_insample_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def barplot_outsample_mse_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='outsample_mse', **kwargs)

def barplot_mse_vs_k(*args, **kwargs):
    return barplot_metric_vs_k(*args, metric='mse', **kwargs)

def barplot_df_vs_k(*args, **kwargs):
    return barplot_metric_vs_k(*args, metric='df', **kwargs)

def barplot_insample_vs_k(*args, **kwargs):
    return barplot_metric_vs_k(*args, metric='insample', **kwargs)

def barplot_outsample_mse_vs_k(*args, **kwargs):
    return barplot_metric_vs_k(*args, metric='outsample_mse', **kwargs)