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
    'plot_outsample_mse_vs_k'
]

class PlottingConfig:
    """Configuration class for plotting parameters."""
    COLORS = {
        'lasso': '#5D4A98',
        # 'ridge': '#ff7f7f',
        'elastic': '#2C858D',
        'bagged_gs': '#74A9CF',
        'smeared_gs': '#B58500',
        # 'base_rgs': '#9467bd',
        # 'base_gs': '#e377c2',
        'original_gs': '#FD8D3C',
        'rgs': '#000000'
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
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    
    # Create specific tick values for the y-axis
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
        
        for exp in range(int(start_exp), int(end_exp) + 1):
            # Add specific values within each decade
            for multiplier in [1, 2.5, 5, 7.5]:
                tick = multiplier * (base ** exp)
                if vmin <= tick <= vmax:
                    ticks.append(tick)
        
        return np.array(ticks)
    
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
    # Remove minor locator
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    
    # Use a simple formatter that shows all values
    formatter = ticker.FuncFormatter(lambda x, pos: "{:.1f}".format(x))
    ax.yaxis.set_major_formatter(formatter)
    
    # Ensure grid lines
    ax.grid(which='major', linestyle='-', alpha=0.3)

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

# Convenience functions
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