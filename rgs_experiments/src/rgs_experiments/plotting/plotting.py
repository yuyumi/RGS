import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from pathlib import Path

__all__ = [
    'plot_mse_by_sigma',
    'plot_df_by_sigma',
    'plot_insample_by_sigma',
    'plot_mse_by_variance_explained',
    'plot_df_by_variance_explained',
    'plot_insample_by_variance_explained',
    'plot_mse_vs_k',
    'plot_df_vs_k',
    'plot_insample_vs_k'
]

class PlottingConfig:
    """Configuration class for plotting parameters."""
    COLORS = {
        'lasso': '#2ca02c',
        'ridge': '#ff7f7f', 
        'elastic': '#17becf',
        'gs': '#e6b3e6',
        'rgs': '#000000'
    }
    
    METRIC_LABELS = {
        'mse': 'Mean Square Error',
        'insample': 'In-sample Error',
        'df': 'Degrees of Freedom',
        'coef_recovery': 'Coefficient Recovery Error',
        'support_recovery': 'Support Recovery Accuracy'
    }
    
    # Methods: Lasso, Ridge, Elastic Net, Greedy Selection, Randomized Greedy Selection
    METHODS = ['lasso', 'ridge', 'elastic', 'gs', 'rgs']
    
    @classmethod
    def get_metric_label(cls, metric: str) -> str:
        """Get formatted label for a given metric."""
        return cls.METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

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
                   label=method.upper())
                   
            if show_std:
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        ax.set_xlabel('Sigma (Noise Level)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.set_title(f'{PlottingConfig.get_metric_label(metric)} by Sigma Level')
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
                   label=method.upper())
                   
            if show_std:
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.set_title(f'{PlottingConfig.get_metric_label(metric)} by PVE')
        ax.legend(loc='upper right' if metric in ['mse', 'insample'] else 'upper left')
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
                      label=f'{method.upper()}')
        
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
                       label=method.upper())
                       
                ax.fill_between(
                    k_values,
                    np.array(k_metrics) - np.array(k_stds),
                    np.array(k_metrics) + np.array(k_stds),
                    color=PlottingConfig.COLORS[method],
                    alpha=0.2
                )
        
        ax.set_yscale('log')
        ax.set_xlabel('k')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.set_title(f'{PlottingConfig.get_metric_label(metric)} vs k (σ = {target_sigma:.2f})')
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

def plot_mse_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def plot_df_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='df', **kwargs)

def plot_insample_by_variance_explained(*args, **kwargs):
    return plot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def plot_mse_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='mse', **kwargs)

def plot_df_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='df', **kwargs)

def plot_insample_vs_k(*args, **kwargs):
    return plot_metric_vs_k(*args, metric='insample', **kwargs)