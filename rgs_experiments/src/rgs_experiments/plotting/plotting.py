import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Optional, List, Dict
from pathlib import Path
from matplotlib.ticker import FuncFormatter

__all__ = [
    'PlottingConfig',
    'setup_plot',
    'load_and_validate_data', 
    'plot_metric'
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
               'elastic', 
                'bagged_gs', 'smeared_gs',
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


# Keep the ManualLogLocator class
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


def setup_plot(figsize=(7, 5), log_y=False, log_x=False, decimal_x=False, 
               y_values=None, padding=0.05, force_zero=True, legend_loc='upper left'):
    """
    Complete plot setup - replaces setup_plot_style(), create_figure(), 
    enhance_log_axis(), format_x_axis_decimal(), and y-limit setting.
    
    Parameters:
    - figsize: figure size
    - log_y/log_x: use log scale 
    - decimal_x: format x-axis with 0.1 intervals (for variance plots)
    - y_values: list of y-values to set limits (optional)
    - padding: padding percentage for y-limits
    - force_zero: force y-axis to start at 0
    - legend_loc: legend position ('upper left', 'upper right', etc.)
    
    Returns: fig, ax ready to plot on
    """
    # Style
    sns.set_style("white")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 
        'axes.labelsize': 12, 'legend.fontsize': 10
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Y-axis
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: f"{x:.1f}" if x >= 0.1 else f"{x:.2f}"
        ))
        ax.set_ylim(bottom=0.0001)
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # Don't set y-limits manually - let matplotlib auto-scale
        # if y_values:
        #     y_min, y_max = min(y_values), max(y_values)
        #     data_range = y_max - y_min
        #     pad = data_range * padding if data_range > 0 else y_max * padding
        #     bottom = 0 if force_zero and y_min >= 0 else y_min - pad
        #     ax.set_ylim(bottom=bottom, top=y_max + pad)
    
    # X-axis
    if log_x:
        ax.set_xscale('log')
    elif decimal_x:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    
    # Grid
    ax.grid(which='major', linestyle='-', alpha=0.3)
    ax.grid(which='minor', linestyle=':', alpha=0.1)
    
    # Store legend location for later use (don't create legend until we have data)
    ax._legend_loc = legend_loc
    print(f"DEBUG LEGEND: Stored legend_loc='{legend_loc}' in ax._legend_loc")
    
    return fig, ax


def load_and_validate_data(
    results_path: Path, 
    metric: str, 
    need_variance: bool = False, 
    need_rie: bool = False, 
    norm_beta: float = 10.0
) -> pd.DataFrame:
    """
    Load CSV data, convert numeric columns, validate methods, and calculate derived metrics.
    FAILS LOUDLY if any data quality issues are found.
    
    Parameters:
    -----------
    results_path : Path
        Path to CSV file
    metric : str  
        The metric to validate (e.g., 'mse', 'rte', 'rie')
    need_variance : bool
        Whether to calculate variance explained column
    need_rie : bool
        Whether to calculate RIE metric from insample data
    norm_beta : float
        Beta norm for variance explained calculation
        
    Returns:
    --------
    pd.DataFrame
        Clean dataframe with numeric types and derived metrics
        
    Raises:
    -------
    FileNotFoundError
        If CSV file doesn't exist
    ValueError  
        If data validation fails (NaNs, no methods, conversion errors)
    """
    
    # 1. Load CSV file
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {results_path}: {str(e)}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {results_path}")
    
    # 2. Convert numeric-looking columns to float64
    numeric_columns = []
    conversion_errors = []
    
    for col in df.columns:
        # Skip obviously non-numeric columns
        if col in ['method', 'algorithm', 'status'] or col.startswith('_'):
            continue
            
        # Try to convert to numeric
        original_dtype = df[col].dtype
        if original_dtype == 'object' or original_dtype == 'string':
            try:
                # First, try to identify what values would fail conversion
                temp_numeric = pd.to_numeric(df[col], errors='coerce')
                failed_conversion = df[col][temp_numeric.isna() & df[col].notna()]
                
                if len(failed_conversion) > 0:
                    failed_indices = failed_conversion.index.tolist()
                    failed_values = failed_conversion.tolist()
                    conversion_errors.append({
                        'column': col,
                        'values': failed_values[:5],  # Show first 5 problematic values
                        'indices': failed_indices[:5],
                        'total_failed': len(failed_conversion)
                    })
                else:
                    # Safe to convert
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    numeric_columns.append(col)
            except Exception as e:
                conversion_errors.append({
                    'column': col,
                    'error': str(e),
                    'values': df[col].head().tolist(),
                    'indices': list(range(min(5, len(df)))),
                    'total_failed': len(df)
                })
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Already numeric, ensure it's float64
            df[col] = df[col].astype('float64')
            numeric_columns.append(col)
    
    # Fail loudly on conversion errors
    if conversion_errors:
        error_msg = "Failed to convert columns to numeric!\n"
        for error in conversion_errors:
            error_msg += f"- Column '{error['column']}': "
            if 'error' in error:
                error_msg += f"Exception: {error['error']}\n"
            else:
                error_msg += f"{error['total_failed']} values failed conversion\n"
                error_msg += f"  Problematic values: {error['values']} at rows {error['indices']}\n"
        error_msg += "Fix your data source before plotting."
        raise ValueError(error_msg)
    
    # 3. Check for NaN values in ALL numeric columns - FAIL LOUDLY
    nan_problems = []
    for col in numeric_columns:
        nan_mask = df[col].isna()
        if nan_mask.any():
            nan_indices = df.index[nan_mask].tolist()
            nan_problems.append({
                'column': col,
                'count': nan_mask.sum(),
                'indices': nan_indices[:10]  # Show first 10 indices
            })
    
    if nan_problems:
        error_msg = "NaN values detected in data!\n"
        for problem in nan_problems:
            error_msg += f"- Column '{problem['column']}': {problem['count']} NaN values at rows {problem['indices']}"
            if problem['count'] > 10:
                error_msg += f" (and {problem['count'] - 10} more)"
            error_msg += "\n"
        error_msg += "Data integrity compromised. Check your simulation results."
        raise ValueError(error_msg)
    
    # 4. Validate methods for the requested metric
    if metric == 'rie' and need_rie:
        # For RIE, we need insample data first, then we'll calculate RIE
        available_methods = get_available_methods(df, 'insample')
        metric_type = 'insample (for RIE calculation)'
    else:
        available_methods = get_available_methods(df, metric)
        metric_type = metric
    
    if not available_methods:
        # Show what metrics ARE available
        available_metrics = set()
        for col in df.columns:
            if '_' in col:
                metric_part = col.split('_')[0]
                available_metrics.add(metric_part)
        
        raise ValueError(
            f"No data found for metric '{metric}' in {results_path.name}\n"
            f"Available metrics: {sorted(list(available_metrics))}\n"
            f"Available methods: {PlottingConfig.METHODS}"
        )
    
    # 5. Check if all data for available methods is completely empty
    all_empty = True
    empty_methods = []
    
    for method in available_methods:
        if metric == 'rie' and need_rie:
            metric_col = f'insample_{method}'
        else:
            metric_col = f'{metric}_{method}'
            
        if metric_col in df.columns:
            if not df[metric_col].isna().all():
                all_empty = False
            else:
                empty_methods.append(method)
    
    if all_empty:
        raise ValueError(
            f"All data is NaN for metric '{metric_type}'!\n"
            f"Available methods: {available_methods}\n"
            f"All methods have completely empty data. Check simulation output."
        )
    
    # Warn about partially empty methods
    if empty_methods:
        print(f"Warning: Methods with all NaN data for {metric_type}: {empty_methods}")
    
    # 6. Calculate derived metrics
    if need_variance:
        if 'sigma' not in df.columns:
            raise ValueError("Cannot calculate variance explained: 'sigma' column missing")
        # Create a copy to avoid fragmentation warning
        df = df.copy()
        df['var_explained'] = norm_beta / (norm_beta + df['sigma']**2)
    
    if need_rie:
        if metric != 'rie':
            raise ValueError("need_rie=True but metric is not 'rie'")
        
        # Create a copy to avoid fragmentation warning
        df = df.copy()
        
        # Calculate RIE for each method that has insample data
        calculated_any = False
        for method in PlottingConfig.METHODS:
            insample_col = f'insample_{method}'
            if insample_col in df.columns and not df[insample_col].isna().all():
                if 'sigma' not in df.columns:
                    raise ValueError("Cannot calculate RIE: 'sigma' column missing")
                # RIE = (insample/sigma^2) + 1
                df[f'rie_{method}'] = (df[insample_col] / (df['sigma']**2)) + 1
                calculated_any = True
        
        if not calculated_any:
            raise ValueError("Cannot calculate RIE: no valid insample data found for any method")
    
    return df


def plot_metric(
    results_path: Path,
    metric: str,
    x_variable: str = 'sigma',  # 'sigma', 'variance', 'fixed_sigma'
    plot_type: str = 'line',   # 'line', 'bar', 'scatter_line'
    show_std: bool = False,
    log_scale: bool = False,    # For regular plots
    log_scale_x: Optional[bool] = None,  # For fixed_sigma plots
    log_scale_y: Optional[bool] = None,  # For fixed_sigma plots
    save_path: Optional[Path] = None,
    norm_beta: float = 10.0,
    target_sigma: Optional[float] = None,  # For fixed_sigma plots
    sigma_tolerance: float = 1e-6,
    method_filter: Optional[callable] = None  # For fixed_sigma plots
) -> Optional[plt.Figure]:
    """
    Unified plotting function that handles all plot types.
    
    Parameters:
    -----------
    results_path : Path
        Path to CSV file with results
    metric : str
        Metric to plot ('mse', 'rte', 'rie', 'outsample_mse', 'mse_vs_df', 'df_vs_k', 'mse_vs_k')
    x_variable : str
        X-axis variable: 'sigma', 'variance', or 'fixed_sigma'
    plot_type : str
        Plot type: 'line', 'bar', or 'scatter_line'
    show_std : bool
        Whether to show error bars/bands
    log_scale : bool
        Whether to use log scale for y-axis (regular plots only)
    log_scale_x : bool, optional
        Whether to use log scale for x-axis (fixed_sigma plots only)
    log_scale_y : bool, optional  
        Whether to use log scale for y-axis (fixed_sigma plots only)
    save_path : Path, optional
        Path to save figure
    norm_beta : float
        Beta norm for variance explained calculation
    target_sigma : float, optional
        Required for fixed_sigma plots - specific sigma value to filter
    sigma_tolerance : float
        Tolerance for sigma filtering
    method_filter : callable, optional
        Function to filter methods for fixed_sigma plots
        
    Returns:
    --------
    matplotlib.Figure or None
    """
    
    try:
        # Special handling for fixed_sigma plots
        if x_variable == 'fixed_sigma':
            # Auto-generate filename with sigma value if save_path is a directory
            if save_path and save_path.is_dir():
                sigma_str = str(target_sigma).replace('.', '_')
                filename = f"{metric}_sigma_{sigma_str}.png"
                save_path = save_path / filename
            
            return _plot_sigma(
                results_path, metric, target_sigma, save_path, 
                log_scale_x or False, log_scale_y or False, 
                sigma_tolerance, method_filter
            )
        
        # Standard plot logic for other metrics
        need_variance = (x_variable == 'variance')
        need_rie = (metric == 'rie')
        
        df = load_and_validate_data(
            results_path, 
            metric, 
            need_variance=need_variance,
            need_rie=need_rie,
            norm_beta=norm_beta
        )
        
        # Get available methods
        available_methods = get_available_methods(df, metric)
        
        # Setup x-axis configuration
        if x_variable == 'variance':
            x_col = 'var_explained'
            xlabel = 'Proportion of Variance Explained (PVE)'
            decimal_x = True
            # Different legend positions for different metrics
            legend_loc = 'upper right' if metric in ['mse', 'outsample_mse', 'rte', 'rie'] else 'upper left'
        else:  # sigma
            x_col = 'sigma'
            xlabel = 'Sigma (Noise Level)'
            decimal_x = False
            legend_loc = 'upper left'
        
        print(f"DEBUG LEGEND: Metric='{metric}', x_variable='{x_variable}', legend_loc='{legend_loc}'")
        
        # Collect all y-values for axis scaling
        all_y_values = []
        
        # Group data by x-variable
        grouped_data = {}
        for method in available_methods:
            metric_col = f'{metric}_{method}'
            grouped = df.groupby(x_col)[metric_col].agg(['mean', 'std'])
            grouped_data[method] = grouped
            
            # Collect y-values
            all_y_values.extend(grouped['mean'].dropna().tolist())
            if show_std:
                upper = (grouped['mean'] + grouped['std']).dropna()
                lower = (grouped['mean'] - grouped['std']).dropna()
                all_y_values.extend(upper.tolist())
                all_y_values.extend(lower.tolist())
        
        # Setup plot
        fig, ax = setup_plot(
            log_y=log_scale,
            decimal_x=decimal_x,
            legend_loc=legend_loc
        )
        
        print(f"DEBUG LEGEND: After setup_plot, ax._legend_loc = {getattr(ax, '_legend_loc', 'NOT_FOUND')}")
        
        if plot_type == 'line':
            # Line plots
            for method in available_methods:
                grouped = grouped_data[method]
                
                # Main line
                ax.plot(grouped.index, grouped['mean'],
                       marker='o',
                       color=PlottingConfig.COLORS[method],
                       linewidth=1.5,
                       label=PlottingConfig.get_method_label(method))
                
                # Error bars
                if show_std:
                    ax.errorbar(
                        grouped.index,
                        grouped['mean'],
                        yerr=grouped['std'],
                        fmt='none',
                        ecolor=PlottingConfig.COLORS[method],
                        elinewidth=1,
                        capsize=3
                    )
        
        else:  # bar plots
            # Get x-values for bar positioning
            if x_variable == 'variance':
                # Round for grouping - create copy to avoid fragmentation
                df = df.copy()
                df['x_rounded'] = np.round(df[x_col], 2)
                x_values = sorted(df['x_rounded'].unique())
            else:
                x_values = sorted(df[x_col].unique())
            
            n_methods = len(available_methods)
            n_x = len(x_values)
            bar_width = 0.8 / n_methods
            
            # Sort methods (non-GS first, then GS methods)
            gs_methods = [m for m in available_methods if 'gs' in m.lower() or m.lower() == 'rgs']
            non_gs_methods = [m for m in available_methods if m not in gs_methods]
            sorted_methods = sorted(non_gs_methods) + sorted(gs_methods)
            
            for i, method in enumerate(sorted_methods):
                means = []
                stds = []
                positions = []
                
                for j, x_val in enumerate(x_values):
                    if x_variable == 'variance':
                        x_data = df[np.isclose(df['x_rounded'], x_val)]
                    else:
                        x_data = df[df[x_col] == x_val]
                    
                    metric_col = f'{metric}_{method}'
                    means.append(x_data[metric_col].mean())
                    stds.append(x_data[metric_col].std())
                    positions.append(j + (i - n_methods/2 + 0.5) * bar_width)
                
                # Create bars
                ax.bar(
                    positions,
                    means,
                    width=bar_width,
                    color=PlottingConfig.COLORS[method],
                    label=PlottingConfig.get_method_label(method),
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Error bars
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
            
            # Set x-ticks for bars
            ax.set_xticks(range(n_x))
            if x_variable == 'variance':
                ax.set_xticklabels([f"{x:.2f}" for x in x_values])
            else:
                ax.set_xticklabels([f"{x:.2f}" for x in x_values])
        
        # Set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        
        # Always add legend - even for edge cases where points might not be visible
        stored_legend_loc = getattr(ax, '_legend_loc', 'NOT_FOUND')
        final_legend_loc = getattr(ax, '_legend_loc', 'upper left')
        print(f"DEBUG LEGEND: Before legend creation - stored='{stored_legend_loc}', final='{final_legend_loc}'")
        ax.legend(loc=final_legend_loc)
        
        # Save or return
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig
        
    except Exception as e:
        print(f"Error plotting {metric} by {x_variable}: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def _plot_sigma(results_path, metric, target_sigma, save_path, 
               log_scale_x, log_scale_y, sigma_tolerance, method_filter):
    """Internal function to handle fixed_sigma plots (mse_vs_df, df_vs_k, mse_vs_k)."""
    df = pd.read_csv(results_path)
    
    if target_sigma is None:
        raise ValueError("target_sigma is required for fixed_sigma plots")
    
    # Filter by target sigma
    df = df[np.abs(df['sigma'] - target_sigma) < sigma_tolerance]
    if df.empty:
        print(f"No data found for σ = {target_sigma}")
        return None
    
    # Average across simulations
    df_avg = df.groupby('sigma').mean().reset_index()
    
    # Determine what we're plotting
    if metric == 'mse_vs_df':
        x_prefix, y_prefix = 'mse_by_k', 'df_by_k'
        xlabel, ylabel = 'Mean Square Error (MSE)', 'Degrees of Freedom (DF)'
    elif metric == 'df_vs_k':
        x_prefix, y_prefix = 'k', 'df_by_k'  # x will be k values directly
        xlabel, ylabel = 'K (Number of Selected Features)', 'Degrees of Freedom (DF)'
    elif metric == 'mse_vs_k':
        x_prefix, y_prefix = 'k', 'mse_by_k'  # x will be k values directly
        xlabel, ylabel = 'K (Number of Selected Features)', 'Mean Square Error (MSE)'
    else:
        raise ValueError(f"Unknown fixed_sigma metric: {metric}")
    
    # Create figure
    fig, ax = setup_plot(log_x=log_scale_x, log_y=log_scale_y)
    
    # Find available methods with k-specific data
    available_methods = []
    for method in PlottingConfig.METHODS:
        if metric == 'mse_vs_df':
            # Need both MSE and DF columns
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            df_cols = [col for col in df_avg.columns if col.startswith(f'df_by_k_{method}_')]
            if mse_cols and df_cols:
                available_methods.append(method)
        else:
            # Need only one type of column
            y_cols = [col for col in df_avg.columns if col.startswith(f'{y_prefix}_{method}_')]
            if y_cols:
                available_methods.append(method)
    
    # Apply method filter
    if method_filter:
        available_methods = [m for m in available_methods if method_filter(m)]
    
    if not available_methods:
        print(f"No methods with k-specific data found for {metric}")
        return None
    
    # Method markers
    method_markers = {
        'lasso': 'o', 'elastic': 's', 'bagged_gs': 'v', 
        'smeared_gs': '^', 'original_gs': 'D', 'rgs': '*'
    }
    
    # Plot each method's k progression
    for method in available_methods:
        if metric == 'mse_vs_df':
            # MSE vs DF: get both MSE and DF values
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            k_values = sorted([int(col.split('_')[-1]) for col in mse_cols])
            
            x_values, y_values = [], []
            for k in k_values:
                mse_col = f'mse_by_k_{method}_{k}'
                df_col = f'df_by_k_{method}_{k}'
                
                if mse_col in df_avg.columns and df_col in df_avg.columns:
                    mse_val = df_avg[mse_col].values[0]
                    df_val = df_avg[df_col].values[0]
                    
                    if not np.isnan(mse_val) and not np.isnan(df_val):
                        x_values.append(mse_val)
                        y_values.append(df_val)
        
        else:
            # DF vs K or MSE vs K: x is k values, y is metric values
            y_cols = [col for col in df_avg.columns if col.startswith(f'{y_prefix}_{method}_')]
            k_values = sorted([int(col.split('_')[-1]) for col in y_cols])
            
            x_values, y_values = [], []
            for k in k_values:
                y_col = f'{y_prefix}_{method}_{k}'
                
                if y_col in df_avg.columns:
                    y_val = df_avg[y_col].values[0]
                    
                    if not np.isnan(y_val):
                        x_values.append(k)  # x is just k value
                        y_values.append(y_val)
        
        if x_values and y_values:
            marker = method_markers.get(method, 'o')
            
            # Plot line and scatter
            ax.plot(x_values, y_values, '-', 
                   color=PlottingConfig.COLORS[method], linewidth=1.5,
                   label=PlottingConfig.get_method_label(method))
            
            ax.scatter(x_values, y_values, 
                      color=PlottingConfig.COLORS[method], marker=marker,
                      s=60, zorder=3, edgecolors='black', linewidths=0.5)
# Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Always add legend
    ax.legend(loc='best')
    
    # Save or return
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig