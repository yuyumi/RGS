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
    'plot_outsample_mse_by_sigma',
    'plot_mse_by_variance_explained',
    'plot_outsample_mse_by_variance_explained',
    'plot_coef_recovery_by_variance_explained',
    'plot_rte_by_variance_explained',
    'plot_rie_by_sigma',
    'plot_rie_by_variance_explained',
    'barplot_metric_by_sigma',
    'barplot_metric_by_variance_explained',
    'barplot_mse_by_sigma',
    'barplot_outsample_mse_by_sigma',
    'barplot_mse_by_variance_explained',
    'barplot_insample_by_variance_explained',
    'barplot_outsample_mse_by_variance_explained',
    'barplot_rte_by_variance_explained',
    'barplot_rte_by_sigma',
    'barplot_rie_by_sigma',
    'barplot_rie_by_variance_explained',
    'plot_mse_vs_df_by_k'
]

class PlottingConfig:
    """Configuration class for plotting parameters."""
    COLORS = {
        'lasso': '#1A5276',
        'ridge': '#2980B9',
        'elastic': '#7FB3D5',
        'bagged_gs': '#922B21',
        'smeared_gs': '#CB4335',
        'base_gs': '#E74C3C',
        'original_gs': '#D35400',
        'rgs': '#F8C471'
    }
    
    METRIC_LABELS = {
        'mse': 'Mean Square Error',
        'insample': 'In-sample Error',
        'outsample_mse': 'Out-of-sample Mean Square Error',
        'df': 'Degrees of Freedom',
        'coef_recovery': 'Coefficient Recovery Error',
        'support_recovery': 'Support Recovery Accuracy',
        'rte': 'Relative Test Error',
        'rie': 'Relative In-sample Error'
    }
    
    METHODS = ['lasso', 'elastic', 'bagged_gs', 'smeared_gs', 'original_gs', 'rgs']
    
    METHOD_LABELS = {
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'elastic': 'Elastic',
        'bagged_gs': 'Bagged GS',
        'smeared_gs': 'Smeared GS',
        'base_rgs': 'Base RGS',
        'base_gs': 'Base GS',
        'original_gs': 'GS',
        'rgs': 'RGS'
    }
    
    @classmethod
    def get_metric_label(cls, metric: str) -> str:
        return cls.METRIC_LABELS.get(metric, metric.replace('_', ' ').title())
    
    @classmethod
    def get_method_label(cls, method: str) -> str:
        return cls.METHOD_LABELS.get(method, method.upper())

class ManualLogLocator(ticker.Locator):
    def __init__(self, tick_function):
        self.tick_function = tick_function
    
    def tick_values(self, vmin, vmax):
        return self.tick_function(vmin, vmax)
    
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

def get_available_methods(df: pd.DataFrame, metric_prefix: str) -> List[str]:
    """Get list of available methods based on column prefixes."""
    return [m for m in PlottingConfig.METHODS if f'{metric_prefix}_{m}' in df.columns]

def setup_plot_style():
    """Set up consistent plot styling."""
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })

def create_figure():
    """Create and return a new figure with standard size."""
    return plt.subplots(figsize=(7, 5))

def enhance_log_axis(ax):
    """Apply consistent log axis formatting."""
    def specific_log_ticks(vmin, vmax):
        if vmin <= 0:
            vmin = 1e-10
        
        vmin_log = np.log(vmin) / np.log(10)
        vmax_log = np.log(vmax) / np.log(10)
        start_exp = np.floor(vmin_log)
        end_exp = np.ceil(vmax_log)
        
        ticks = []
        for exp in range(int(start_exp), int(end_exp) + 1):
            decade = 10 ** exp
            ticks.append(decade)
            if exp < end_exp:
                for multiplier in [2, 5]:
                    tick = multiplier * decade
                    if vmin <= tick <= vmax:
                        ticks.append(tick)
        return np.array(sorted(ticks))
    
    ax.yaxis.set_major_locator(ManualLogLocator(specific_log_ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f"{x:.1f}" if x >= 0.1 else f"{x:.2f}"
    ))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax.grid(which='major', linestyle='-', alpha=0.3)
    ax.grid(which='minor', linestyle=':', alpha=0.1)

def format_x_axis_decimal(ax):
    """Format x-axis to show values at 0.1 intervals."""
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(which='major', axis='x', linestyle='-', alpha=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', alpha=0.2)

def _load_and_prepare_data(results_path: Path, metric: str, need_variance: bool = False, norm_beta: float = 10.0):
    """Load CSV data and prepare it for plotting."""
    df = pd.read_csv(results_path)
    df = df.apply(pd.to_numeric, errors='ignore')
    if 'method' in df.columns:
        df = df.drop('method', axis=1)
    
    # Add SNR calculation (same as original)
    norm_beta_squared = 10.0  # β^T Σβ = 10 based on setup
    df['snr'] = norm_beta_squared / (df['sigma']**2)
    
    # Calculate RIE if needed
    if metric == 'rie':
        for method in PlottingConfig.METHODS:
            insample_col = f'insample_{method}'
            if insample_col in df.columns and not df[insample_col].isna().all():
                df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1
    
    # Calculate variance explained if needed
    if need_variance:
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
    
    return df

def _setup_axis_scaling(ax, log_scale: bool, all_means: List, all_upper_bounds: List = None):
    """Setup axis scaling and formatting."""
    if log_scale:
        ax.set_yscale('log')
        enhance_log_axis(ax)
    else:
        ax.set_yscale('linear')
        if all_means:
            max_value = max(all_upper_bounds) if all_upper_bounds else max(all_means)
            min_value = min(all_means)
            data_range = max_value - min_value
            padding = data_range * 0.05 if data_range > 0 else max_value * 0.05
            y_min = max(0, min_value - padding) if min_value >= 0 else min_value - padding
            y_max = max_value + padding
            ax.set_ylim(bottom=y_min, top=y_max)
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
        ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)

def plot_metric_by_sigma(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                        show_std: bool = False, log_scale: bool = False) -> Optional[plt.Figure]:
    """Plot metric vs sigma with error bars."""
    try:
        df = _load_and_prepare_data(results_path, metric)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods or not any(not df[f'{metric}_{m}'].isna().all() for m in available_methods):
            print(f"No data found for metric '{metric}' in {Path(results_path).name}")
            return None
        
        fig, ax = create_figure()
        setup_plot_style()
        
        all_means, all_upper_bounds = [], []
        
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
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           fmt='none', ecolor=PlottingConfig.COLORS[method],
                           elinewidth=1, capsize=3)
        
        _setup_axis_scaling(ax, log_scale, all_means, all_upper_bounds if show_std else None)
        
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

def plot_metric_by_variance_explained(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                                    show_std: bool = True, log_scale: bool = False, norm_beta: float = 10.0) -> Optional[plt.Figure]:
    """Plot metric vs proportion of variance explained with error bars."""
    try:
        df = _load_and_prepare_data(results_path, metric, need_variance=True, norm_beta=norm_beta)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
        
        fig, ax = create_figure()
        setup_plot_style()
        
        all_means, all_upper_bounds = [], []
        
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
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                           fmt='none', ecolor=PlottingConfig.COLORS[method],
                           elinewidth=1, capsize=3)
        
        _setup_axis_scaling(ax, log_scale, all_means, all_upper_bounds if show_std else None)
        format_x_axis_decimal(ax)
        
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rte', 'rie'] else 'upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by variance explained: {str(e)}")
        plt.close()
        return None

def plot_mse_vs_df_by_k(results_path: Path, target_sigma: float, save_path: Optional[Path] = None,
                       show_std: bool = False, log_scale_mse: bool = False, log_scale_df: bool = False,
                       sigma_tolerance: float = 1e-6, method_filter: Optional[callable] = None) -> Optional[plt.Figure]:
    """Plot MSE vs DF for each method's k progression at target sigma."""
    try:
        df = pd.read_csv(results_path)
        # Convert columns to numeric, keeping non-numeric columns as-is
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # Keep original values if conversion fails
        if 'method' in df.columns:
            df = df.drop('method', axis=1)
        
        df = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
        
        if df.empty:
            print(f"No data found for σ = {target_sigma}")
            return None
        
        df_avg = df.groupby('sigma').mean().reset_index()
        available_methods = []
        
        # Explicitly filter to only original_gs and rgs methods for MSE vs DF by k plots
        target_methods = ['original_gs', 'rgs']
        for method in target_methods:
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            df_cols = [col for col in df_avg.columns if col.startswith(f'df_by_k_{method}_')]
            if mse_cols and df_cols:
                available_methods.append(method)
        
        if method_filter:
            available_methods = [m for m in available_methods if method_filter(m)]
        
        if not available_methods:
            print(f"No methods with k-specific data found")
            return None
        
        fig, ax = create_figure()
        setup_plot_style()
        
        method_markers = {'lasso': 'o', 'elastic': 's', 'bagged_gs': 'v', 
                         'smeared_gs': '^', 'original_gs': 'D', 'rgs': '*'}
        
        for method in available_methods:
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            k_values = sorted([int(col.split('_')[-1]) for col in mse_cols if int(col.split('_')[-1]) > 0])
            
            mse_values, df_values = [], []
            for k in k_values:
                mse_col = f'mse_by_k_{method}_{k}'
                df_col = f'df_by_k_{method}_{k}'
                
                if mse_col in df_avg.columns and df_col in df_avg.columns:
                    mse_val = df_avg[mse_col].values[0]
                    df_val = df_avg[df_col].values[0]
                    
                    if not np.isnan(mse_val) and not np.isnan(df_val):
                        mse_values.append(mse_val)
                        df_values.append(df_val)
            
            if mse_values and df_values:
                marker = method_markers.get(method, 'o')
                ax.plot(mse_values, df_values, '-', color=PlottingConfig.COLORS[method],
                       linewidth=1.5, label=PlottingConfig.get_method_label(method))
                ax.scatter(mse_values, df_values, color=PlottingConfig.COLORS[method],
                          marker=marker, s=60, zorder=3, edgecolors='black', linewidths=0.5)
                
                # Add k labels for k=1 and multiples of 5
                for i, k in enumerate(k_values[:len(mse_values)]):
                    if k == 1:
                        label = 'k=1'
                    elif k % 5 == 0:
                        label = str(k)
                    else:
                        continue
                    
                    ax.annotate(
                        label,
                        (mse_values[i], df_values[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='none')
                    )
        
        # Collect all data points for axis limit calculations
        all_mse_values = []
        all_df_values = []
        for method in available_methods:
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            k_values = sorted([int(col.split('_')[-1]) for col in mse_cols if int(col.split('_')[-1]) > 0])
            
            for k in k_values:
                mse_col = f'mse_by_k_{method}_{k}'
                df_col = f'df_by_k_{method}_{k}'
                
                if mse_col in df_avg.columns and df_col in df_avg.columns:
                    mse_val = df_avg[mse_col].values[0]
                    df_val = df_avg[df_col].values[0]
                    
                    if not np.isnan(mse_val) and not np.isnan(df_val):
                        all_mse_values.append(mse_val)
                        all_df_values.append(df_val)

        # Set axis scales
        if log_scale_mse:
            ax.set_xscale('log')
        else:
            # Set x-axis limits with padding to ensure markers are visible
            if all_mse_values:
                x_min, x_max = min(all_mse_values), max(all_mse_values)
                x_range = x_max - x_min
                x_padding = x_range * 0.08 if x_range > 0 else x_max * 0.08
                ax.set_xlim(left=x_min - x_padding, right=x_max + x_padding)
        
        if log_scale_df:
            ax.set_yscale('log')
        else:
            # Set y-axis limits with padding to ensure markers are visible  
            if all_df_values:
                y_min, y_max = min(all_df_values), max(all_df_values)
                y_range = y_max - y_min
                y_padding = y_range * 0.08 if y_range > 0 else y_max * 0.08
                ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
        
        ax.set_xlabel('Mean Square Error (MSE)')
        ax.set_ylabel('Degrees of Freedom (DF)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error plotting MSE vs DF: {str(e)}")
        plt.close()
        return None

def barplot_metric_by_sigma(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                           show_std: bool = True, log_scale: bool = False) -> Optional[plt.Figure]:
    """Create bar plot of metric by sigma."""
    try:
        df = _load_and_prepare_data(results_path, metric)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
        
        fig, ax = create_figure()
        setup_plot_style()
        
        snr_values = sorted(df['snr'].unique())
        n_methods = len(available_methods)
        n_snr = len(snr_values)
        bar_width = 0.8 / n_methods
        
        # Maintain the order from PlottingConfig.METHODS (same as original)
        sorted_methods = [m for m in PlottingConfig.METHODS if m in available_methods]
        
        for i, method in enumerate(sorted_methods):
            means, stds, positions = [], [], []
            
            for j, snr in enumerate(snr_values):
                snr_data = df[df['snr'] == snr]
                metric_col = f'{metric}_{method}'
                means.append(snr_data[metric_col].mean())
                stds.append(snr_data[metric_col].std())
                positions.append(j + (i - n_methods/2 + 0.5) * bar_width)
            
            ax.bar(positions, means, width=bar_width,
                   color=PlottingConfig.COLORS[method],
                   label=PlottingConfig.get_method_label(method),
                   edgecolor='black', linewidth=0.5)
            
            if show_std:
                ax.errorbar(positions, means, yerr=stds, fmt='none',
                           color='black', capsize=3, linewidth=1, capthick=1)
        
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        ax.set_xticks(range(n_snr))
        ax.set_xticklabels([f"{snr:.2f}" for snr in snr_values])
        ax.set_xlabel('SNR')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rie'] else 'upper left')
        ax.grid(True, axis='y', alpha=0.3)

        # Calculate appropriate y-limits from the data, including error bars if shown
        all_values = []
        for method in sorted_methods:
            metric_col = f'{metric}_{method}'
            for snr in snr_values:
                snr_data = df[df['snr'] == snr]
                mean = snr_data[metric_col].mean()
                if show_std:
                    std = snr_data[metric_col].std()
                    all_values.extend([mean - std, mean + std])
                else:
                    all_values.append(mean)

        if all_values:
            min_val = min(all_values)
            max_val = max(all_values) 
            data_range = max_val - min_val
            padding = data_range * 0.05
            ax.set_ylim(bottom=min_val - padding, top=max_val + padding)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by sigma: {str(e)}")
        plt.close()
        return None

def barplot_metric_by_variance_explained(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                                       show_std: bool = True, log_scale: bool = True, norm_beta: float = 10.0) -> Optional[plt.Figure]:
    """Create bar plot of metric by variance explained."""
    try:
        df = _load_and_prepare_data(results_path, metric, need_variance=True, norm_beta=norm_beta)
        available_methods = get_available_methods(df, metric)
        
        if not available_methods:
            raise ValueError(f"No {metric} data found for any method")
        
        fig, ax = create_figure()
        setup_plot_style()
        
        # Round var_explained for grouping
        df['var_explained_rounded'] = np.round(df['var_explained'], 2)
        var_explained_values = sorted(df['var_explained_rounded'].unique())
        
        n_methods = len(available_methods)
        bar_width = 0.8 / n_methods
        
        for i, method in enumerate(available_methods):
            means, stds, positions = [], [], []
            
            for j, var_expl in enumerate(var_explained_values):
                var_data = df[np.isclose(df['var_explained_rounded'], var_expl)]
                metric_col = f'{metric}_{method}'
                means.append(var_data[metric_col].mean())
                stds.append(var_data[metric_col].std())
                positions.append(j + (i - n_methods/2 + 0.5) * bar_width)
            
            ax.bar(positions, means, width=bar_width,
                   color=PlottingConfig.COLORS[method],
                   label=PlottingConfig.get_method_label(method),
                   edgecolor='black', linewidth=0.5)
            
            if show_std:
                ax.errorbar(positions, means, yerr=stds, fmt='none',
                           color='black', capsize=3, linewidth=1, capthick=1)
        
        if log_scale:
            ax.set_yscale('log')
            enhance_log_axis(ax)
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(which='minor', axis='y', linestyle=':', alpha=0.2)
        
        ax.set_xticks(range(len(var_explained_values)))
        ax.set_xticklabels([f"{var:.2f}" for var in var_explained_values])
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rie'] else 'upper left')
        ax.grid(True, axis='y', alpha=0.3)

        # Calculate appropriate y-limits for ALL metrics, including error bars if shown
        if not log_scale:
            all_values = []
            for method in available_methods:
                metric_col = f'{metric}_{method}'
                for var_expl in var_explained_values:
                    var_data = df[np.isclose(df['var_explained_rounded'], var_expl)]
                    mean = var_data[metric_col].mean()
                    if show_std:
                        std = var_data[metric_col].std()
                        all_values.extend([mean - std, mean + std])
                    else:
                        all_values.append(mean)
            
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
            return None
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by variance explained: {str(e)}")
        plt.close()
        return None

# Convenience functions - maintain exact same API
def plot_mse_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_sigma(*args, metric='mse', **kwargs)

def plot_insample_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_sigma(*args, metric='insample', **kwargs)

def plot_rie_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_sigma(*args, metric='rie', **kwargs)

def plot_outsample_mse_by_sigma(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_sigma(*args, metric='outsample_mse', **kwargs)

def plot_mse_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='mse', **kwargs)

def plot_insample_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='insample', **kwargs)

def plot_rie_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='rie', **kwargs)

def plot_outsample_mse_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='outsample_mse', **kwargs)

def plot_coef_recovery_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='coef_recovery', **kwargs)

def plot_rte_by_variance_explained(*args, **kwargs):
    kwargs.setdefault('log_scale', False)
    return plot_metric_by_variance_explained(*args, metric='rte', **kwargs)

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
    return barplot_metric_by_variance_explained(*args, metric='rte', **kwargs) 