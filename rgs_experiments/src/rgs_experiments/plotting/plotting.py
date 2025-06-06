import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Optional, List, Dict
from pathlib import Path
from matplotlib.ticker import FuncFormatter

__all__ = [
    'plot_mse_by_sigma',
    # 'plot_insample_by_sigma', # Commented out
    'plot_outsample_mse_by_sigma',
    'plot_mse_by_variance_explained',
    # 'plot_insample_by_variance_explained', # Commented out
    'plot_outsample_mse_by_variance_explained',
    'plot_coef_recovery_by_variance_explained',
    'plot_rte_by_variance_explained',
    'plot_rie_by_sigma',  # New function
    'plot_rie_by_variance_explained',  # New function
    # New bar plot functions
    'barplot_metric_by_sigma',
    'barplot_metric_by_variance_explained',
    'barplot_mse_by_sigma',
    # 'barplot_insample_by_sigma', # Commented out
    'barplot_outsample_mse_by_sigma',
    'barplot_mse_by_variance_explained',
    'barplot_insample_by_variance_explained', # Commented out
    'barplot_outsample_mse_by_variance_explained',
    'barplot_rte_by_variance_explained',
    'barplot_rte_by_sigma',
    'barplot_rie_by_sigma',  # New function
    'barplot_rie_by_variance_explained',  # New function
    'plot_mse_vs_df_by_k'
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
        'original_gs': '#D35400',   # Orange
        'rgs': '#F8C471'          # Light orange
    }
    
    METRIC_LABELS = {
        'mse': 'Mean Square Error',
        'insample': 'In-sample Error',  # Kept for legacy compatibility
        'outsample_mse': 'Out-of-sample Mean Square Error',
        'df': 'Degrees of Freedom',
        'coef_recovery': 'Coefficient Recovery Error',
        'support_recovery': 'Support Recovery Accuracy',
        'rte': 'Relative Test Error',
        'rie': 'Relative In-sample Error'  # Added for RIE
    }
    
    # Methods: Lasso, Ridge, Elastic Net, Greedy Selection, Randomized Greedy Selection
    METHODS = ['lasso', 
            #    'ridge',
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
    show_std: bool = False,
    log_scale: bool = False  # Default to linear scale
) -> Optional[plt.Figure]:
    """Plot metric vs sigma with error bars."""
    try:
        df = pd.read_csv(results_path)
        df = df.apply(pd.to_numeric, errors='ignore')
        if 'method' in df.columns:
            df = df.drop('method', axis=1)
        
        # For RIE, we need to calculate it from insample and sigma
        if metric == 'rie':
            # Calculate RIE for each method that has insample data
            for method in PlottingConfig.METHODS:
                insample_col = f'insample_{method}'
                if insample_col in df.columns and not df[insample_col].isna().all():
                    # Calculate RIE as (insample/sigma^2)+1
                    df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1

        else:
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
        
        # Calculate all means and std values for proper y-axis scaling
        all_means = []
        all_upper_bounds = []  # For error bars if shown
        
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            grouped = df.groupby('sigma')[metric_col].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'],
                marker='o',
                color=PlottingConfig.COLORS[method],
                label=PlottingConfig.get_method_label(method))
                   
            all_means.extend(grouped['mean'].tolist())
            
            if show_std:
                upper_bounds = grouped['mean'] + grouped['std']
                all_upper_bounds.extend(upper_bounds.tolist())
                
                # Add error bars instead of shaded regions
                ax.errorbar(
                    grouped.index,
                    grouped['mean'],
                    yerr=grouped['std'],
                    fmt='none',  # No connecting line
                    ecolor=PlottingConfig.COLORS[method],
                    elinewidth=1,
                    capsize=3
                )
        
        # Apply scale based on parameter
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
            # ax.set_ylim(bottom=0.0001)
        else:
            # Linear scale
            ax.set_yscale('linear')
            
            # Set y limits with 5% padding
            if all_means:  # Make sure we have data
                max_value = max(all_upper_bounds) if show_std and all_upper_bounds else max(all_means)
                min_value = min(all_means)
                
                # Calculate appropriate padding (5% of range)
                data_range = max_value - min_value
                padding = data_range * 0.05 if data_range > 0 else max_value * 0.05
                
                # Set limits with padding, but don't go below 0 unless data does
                y_min = max(0, min_value - padding) if min_value >= 0 else min_value - padding
                y_max = max_value + padding
                
                ax.set_ylim(bottom=y_min, top=y_max)
            
            # Format y-axis for linear scale
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            
            # Add grid for linear scale
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        ax.set_xlabel('Sigma (Noise Level)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
            
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
    log_scale: bool = False,  # Default to linear scale
    norm_beta: float = 10.0
) -> Optional[plt.Figure]:
    """Plot metric vs proportion of variance explained with error bars."""
    try:
        df = pd.read_csv(results_path)
        df = df.apply(pd.to_numeric, errors='ignore')
        if 'method' in df.columns:
            df = df.drop('method', axis=1)
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
        
        fig, ax = create_figure()
        setup_plot_style()

        # For RIE, we need to calculate it from insample and sigma
        if metric == 'rie':
            # Calculate RIE for each method that has insample data
            for method in PlottingConfig.METHODS:
                insample_col = f'insample_{method}'
                if insample_col in df.columns and not df[insample_col].isna().all():
                    # Calculate RIE as (insample/sigma^2)+1
                    df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1
        
        available_methods = get_available_methods(df, metric)
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
        
        # For y-axis scaling
        all_means = []
        all_upper_bounds = []
            
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            grouped = df.groupby('var_explained')[metric_col].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'],
                marker='o',
                color=PlottingConfig.COLORS[method],
                linewidth=1.5,
                label=PlottingConfig.get_method_label(method))
            
            all_means.extend(grouped['mean'].tolist())
                   
            if show_std:
                upper_bounds = grouped['mean'] + grouped['std']
                all_upper_bounds.extend(upper_bounds.tolist())
                
                # Add error bars instead of shaded regions
                ax.errorbar(
                    grouped.index,
                    grouped['mean'],
                    yerr=grouped['std'],
                    fmt='none',  # No connecting line
                    ecolor=PlottingConfig.COLORS[method],
                    elinewidth=1,
                    capsize=3
                )
        
        # Apply scale based on parameter
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
            ax.set_ylim(bottom=0.0001)
        else:
            # Linear scale
            ax.set_yscale('linear')
            
            # Set y limits with 5% padding
            if all_means:  # Make sure we have data
                max_value = max(all_upper_bounds) if show_std and all_upper_bounds else max(all_means)
                min_value = min(all_means)
                
                # Calculate appropriate padding (5% of range)
                data_range = max_value - min_value
                padding = data_range * 0.05 if data_range > 0 else max_value * 0.05
                
                # Set limits with padding, but don't go below 0 unless data does
                y_min = max(0, min_value - padding) if min_value >= 0 else min_value - padding
                y_max = max_value + padding
                
                ax.set_ylim(bottom=y_min, top=y_max)
            
            # Format y-axis for linear scale
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            
            # Add grid for linear scale
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        # Format x-axis with decimal intervals
        format_x_axis_decimal(ax)
        
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rte', 'rie'] else 'upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by variance explained: {str(e)}")
        plt.close()
        return None
    
def plot_mse_vs_df_by_k(
    results_path: Path, 
    target_sigma: float,
    save_path: Optional[Path] = None,
    show_std: bool = False,
    log_scale_mse: bool = False,
    log_scale_df: bool = False,
    sigma_tolerance: float = 1e-6,
    method_filter: Optional[callable] = None  # Filter to exclude certain methods
) -> Optional[plt.Figure]:
    """
    Plot MSE vs DF for each method, showing different k values for a specific sigma value.
    Uses line plots to connect points for each method.
    
    Parameters:
    -----------
    results_path : Path
        Path to the CSV file containing simulation results
    target_sigma : float
        Plot only results for this specific sigma value
    save_path : Path, optional
        Path to save the figure
    show_std : bool
        Whether to show standard deviation as error bars (not used for k plots)
    log_scale_mse : bool
        Whether to use log scale for MSE axis
    log_scale_df : bool
        Whether to use log scale for DF axis
    sigma_tolerance : float
        Tolerance for identifying the target sigma in the data
    method_filter : callable, optional
        Function to filter methods, should take a method name and return True/False
        
    Returns:
    --------
    matplotlib.Figure or None
        The generated figure or None if an error occurred
    """
    try:
        df = pd.read_csv(results_path)
        df = df.apply(pd.to_numeric, errors='ignore')
        if 'method' in df.columns:
            df = df.drop('method', axis=1)
        
        # Filter by target sigma
        df = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
        if df.empty:
            print(f"No data found for σ = {target_sigma} in {Path(results_path).name}")
            return None
        
        # Group data by simulation to average across simulations
        group_cols = ['simulation', 'sigma']
        df_avg = df.groupby('sigma').mean().reset_index()
        
        # Create figure
        fig, ax = create_figure()
        setup_plot_style()
        
        # Identify methods that have k-specific data
        available_methods = []
        for method in PlottingConfig.METHODS:
            # Check if method has k-specific columns
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            df_cols = [col for col in df_avg.columns if col.startswith(f'df_by_k_{method}_')]
            
            if mse_cols and df_cols:
                available_methods.append(method)
        
        # Apply method filter if provided
        if method_filter:
            available_methods = [m for m in available_methods if method_filter(m)]
        
        if not available_methods:
            print(f"No methods with both MSE and DF k-specific data in {Path(results_path).name}")
            return None
        
        # Define a dictionary of markers by method type
        method_markers = {
            'lasso': 'o',
            'elastic': 's',  # square
            'bagged_gs': 'v',  # triangle down
            'smeared_gs': '^',  # triangle up 
            'original_gs': 'D',  # diamond
            'rgs': '*'  # star
        }
        
        # Plot each method's k progression
        for method in available_methods:
            # Extract k-specific columns
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            df_cols = [col for col in df_avg.columns if col.startswith(f'df_by_k_{method}_')]
            
            # Extract k numbers from column names
            k_values = []
            for col in mse_cols:
                k = int(col.split('_')[-1])
                k_values.append(k)
            
            # Sort by k values
            k_values = sorted(k_values)
            
            # Extract MSE and DF values for each k
            mse_values = []
            df_values = []
            
            for k in k_values:
                mse_col = f'mse_by_k_{method}_{k}'
                df_col = f'df_by_k_{method}_{k}'
                
                if mse_col in df_avg.columns and df_col in df_avg.columns:
                    mse_value = df_avg[mse_col].values[0]  # Already averaged across simulations
                    df_value = df_avg[df_col].values[0]
                    
                    # Only add if both values are valid
                    if not np.isnan(mse_value) and not np.isnan(df_value):
                        mse_values.append(mse_value)
                        df_values.append(df_value)
            
            if mse_values and df_values:
                # Use the dictionary to get the marker
                marker = method_markers.get(method, 'o')
                
                # Plot each point with a marker and connect with lines
                ax.plot(
                    mse_values, 
                    df_values,
                    '-',  # Line style
                    color=PlottingConfig.COLORS[method],
                    linewidth=1.5,
                    label=PlottingConfig.get_method_label(method)
                )
                
                # Add markers at each point
                ax.scatter(
                    mse_values, 
                    df_values,
                    color=PlottingConfig.COLORS[method],
                    marker=marker,
                    s=60,
                    zorder=3,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Optionally add k labels for important points
                # Uncomment if you want to show k values
                """
                for i, k in enumerate(k_values[:len(mse_values)]):
                    # Only label some k values to avoid clutter
                    if k % 5 == 0 or k == k_values[-1]:  # Label every 5th k and the last k
                        ax.annotate(
                            f'{k}',
                            (mse_values[i], df_values[i]),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha='center',
                            fontsize=8
                        )
                """
        
        # Apply scales
        if log_scale_mse:
            ax.set_xscale('log')
            
        def format_plain(x, pos):
            return f"{x:.2f}"  # Show as plain decimal with 2 decimal places
            
        ax.xaxis.set_major_formatter(FuncFormatter(format_plain))
        
        if log_scale_df:
            ax.set_yscale('log')
        else:
            # For linear y-scale, set reasonable limits
            if any(df_values):
                y_max = max(df_values) * 1.05
                y_min = min(df_values) * 0.95
                ax.set_ylim(bottom=y_min, top=y_max)
                
        # Set labels
        ax.set_xlabel('Training Error')
        ax.set_ylabel('Degrees of Freedom (DF)')
        
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
        print(f"Error plotting MSE vs DF by k: {str(e)}")
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
        df = df.apply(pd.to_numeric, errors='ignore')
        if 'method' in df.columns:
            df = df.drop('method', axis=1)

        # For RIE, we need to calculate it from insample and sigma
        if metric == 'rie':
            # Calculate RIE for each method that has insample data
            for method in PlottingConfig.METHODS:
                insample_col = f'insample_{method}'
                if insample_col in df.columns and not df[insample_col].isna().all():
                    # Calculate RIE as (insample/sigma^2)+1
                    df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1
        
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
            ax.set_ylim(bottom=0.0001)
        else:
            
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
        df = df.apply(pd.to_numeric, errors='ignore')
        if 'method' in df.columns:
            df = df.drop('method', axis=1)
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)

        # For RIE, we need to calculate it from insample and sigma
        if metric == 'rie':
            # Calculate RIE for each method that has insample data
            for method in PlottingConfig.METHODS:
                insample_col = f'insample_{method}'
                if insample_col in df.columns and not df[insample_col].isna().all():
                    # Calculate RIE as (insample/sigma^2)+1
                    df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1
        
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
            # For linear scale or for RTE
            if metric != 'rte':
                # For non-RTE metrics, start from 0
                ax.set_ylim(bottom=0)
            
            # Add proper y-tick formatting for linear scale
            formatter = ticker.FormatStrFormatter('%.3f')
            ax.yaxis.set_major_formatter(formatter)
            
            # Set major ticks at reasonable intervals
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            
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
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rie'] else 'upper left')
        ax.grid(True, axis='y', alpha=0.3)

        if metric == 'rte' and not log_scale:
            # Get all plotted values including error bars
            all_values = []
            for method in available_methods:
                metric_col = f'{metric}_{method}'
                for var_expl in var_explained_values:
                    var_data = df[np.isclose(df['var_explained_rounded'], var_expl)][metric_col]
                    mean = var_data.mean()
                    std = var_data.std() if show_std else 0
                    all_values.extend([mean - std, mean + std])
            
            # Set limits with 5% padding
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                
                # Add padding (5% of range)
                data_range = y_max - y_min
                padding = data_range * 0.05
                
                ax.set_ylim(bottom=y_min - padding, top=y_max + padding)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by variance explained: {str(e)}")
        plt.close()
        return None

# Update the convenience functions to use linear scale by default
def plot_mse_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_sigma(*args, metric='mse', **kwargs)

def plot_insample_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_sigma(*args, metric='insample', **kwargs)

def plot_rie_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_sigma(*args, metric='rie', **kwargs)

def plot_outsample_mse_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_sigma(*args, metric='outsample_mse', **kwargs)

def plot_mse_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def plot_insample_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def plot_rie_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='rie', **kwargs)

def plot_outsample_mse_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='outsample_mse', **kwargs)

def plot_coef_recovery_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='coef_recovery', **kwargs)

def plot_rte_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)  # Set default to False for linear scale
    return plot_metric_by_variance_explained(*args, metric='rte', **kwargs)

# Convenience functions for bar plots
def barplot_mse_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='mse', **kwargs)

def barplot_insample_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='insample', **kwargs)

def barplot_rie_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='rie', **kwargs)

def barplot_rte_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='rte', **kwargs)

def barplot_outsample_mse_by_sigma(*args, **kwargs):
    return barplot_metric_by_sigma(*args, metric='outsample_mse', **kwargs)

def barplot_mse_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def barplot_insample_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def barplot_rie_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='rie', **kwargs)

def barplot_outsample_mse_by_variance_explained(*args, **kwargs):
    return barplot_metric_by_variance_explained(*args, metric='outsample_mse', **kwargs)

def barplot_rte_by_variance_explained(*args, **kwargs):
    """Create bar plot of RTE by variance explained."""
    return barplot_metric_by_variance_explained(*args, metric='rte', **kwargs)