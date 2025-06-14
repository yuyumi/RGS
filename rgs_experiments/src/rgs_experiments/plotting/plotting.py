import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from ..utils.snr_utils import get_signal_strength_from_results, compute_snr, compute_variance_explained

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
    'plot_mse_vs_df_by_k',
    'collect_global_y_limits'
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
        'gs': '#D35400',  # Same color as original_gs since they're the same method
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
    
    METHODS = ['lasso', 'elastic', 'bagged_gs', 'smeared_gs', 'original_gs', 'gs', 'rgs']
    
    METHOD_LABELS = {
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'elastic': 'Elastic',
        'bagged_gs': 'Bagged GS',
        'smeared_gs': 'Smeared GS',
        'base_rgs': 'Base RGS',
        'base_gs': 'Base GS',
        'original_gs': 'GS',
        'gs': 'GS',  # Same label as original_gs
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

def _ensure_pdf_extension(save_path: Path) -> Path:
    """Ensure the save path has a .pdf extension."""
    if save_path.suffix.lower() != '.pdf':
        return save_path.with_suffix('.pdf')
    return save_path

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

def _find_params_file(results_path: Path) -> str:
    """Find the corresponding parameter file for a results file."""
    results_path = Path(results_path)
    
    # Extract the timestamp from the results filename
    # Expected format: simulation_results_<type>_<timestamp>.csv
    filename = results_path.stem
    if 'simulation_results_' in filename:
        # Extract everything after 'simulation_results_'
        suffix = filename.replace('simulation_results_', '')
        params_filename = f"simulation_params_{suffix}.json"
    else:
        # Fallback: try to guess based on timestamp pattern
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            # Look for any params file with this timestamp
            params_pattern = f"*params*{timestamp}*.json"
            params_files = list(results_path.parent.glob(params_pattern))
            if params_files:
                return str(params_files[0])
        
        raise ValueError(f"Could not find parameter file for results file: {results_path}")
    
    params_path = results_path.parent / params_filename
    if not params_path.exists():
        raise ValueError(f"Parameter file not found: {params_path}")
    
    return str(params_path)

def _load_and_prepare_data(results_path: Path, metric: str, need_variance: bool = False):
    """Load CSV data and prepare it for plotting."""
    df = pd.read_csv(results_path)
    # Convert columns to numeric, keeping non-numeric columns as-is
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep original values if conversion fails
    if 'method' in df.columns:
        df = df.drop('method', axis=1)
    
    # Prepare all new columns at once to avoid fragmentation
    new_columns = {}
    
    # Get signal strength using proper utilities - no fallback allowed
    # Try to find the corresponding parameter file
    params_file_path = _find_params_file(results_path)
    signal_strength = get_signal_strength_from_results(df, method="from_params", params_file_path=params_file_path)
    if isinstance(signal_strength, np.ndarray):
        # If we have per-row signal strengths, use them
        new_columns['signal_strength'] = signal_strength
        new_columns['snr'] = [compute_snr(ss, sigma) for ss, sigma in zip(signal_strength, df['sigma'])]
        if need_variance:
            new_columns['var_explained'] = [compute_variance_explained(ss, sigma) for ss, sigma in zip(signal_strength, df['sigma'])]
    else:
        # Single signal strength value for all rows
        new_columns['signal_strength'] = signal_strength
        new_columns['snr'] = df['sigma'].apply(lambda sigma: compute_snr(signal_strength, sigma))
        if need_variance:
            new_columns['var_explained'] = df['sigma'].apply(lambda sigma: compute_variance_explained(signal_strength, sigma))
    
    # Check if RIE columns exist, if not, construct them from insample error
    # RIE = (insample error)/sigma^2 + 1
    available_methods = get_available_methods(df, 'insample')
    
    for method in available_methods:
        rie_col = f'rie_{method}'
        insample_col = f'insample_{method}'
        
        if rie_col not in df.columns and insample_col in df.columns:
            # Construct RIE from insample error: RIE = (insample error)/sigma^2 + 1
            new_columns[rie_col] = (df[insample_col] / (df['sigma'] ** 2)) + 1

    
    # Add all new columns at once using concat to avoid fragmentation
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    
    return df

def _setup_axis_scaling(ax, log_scale: bool, all_means: List, all_upper_bounds: List = None, 
                        global_ylim: Optional[Tuple[float, float]] = None):
    """Setup axis scaling and formatting."""
    if log_scale:
        ax.set_yscale('log')
        enhance_log_axis(ax)
        # Apply global limits if provided
        if global_ylim:
            ax.set_ylim(bottom=global_ylim[0], top=global_ylim[1])
    else:
        ax.set_yscale('linear')
        
        # Use global limits if provided, otherwise calculate from data
        if global_ylim:
            ax.set_ylim(bottom=global_ylim[0], top=global_ylim[1])
        elif all_means:
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
                        show_std: bool = False, log_scale: bool = False, 
                        global_ylim: Optional[Tuple[float, float]] = None) -> Optional[plt.Figure]:
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
                # Only show error bars if std is not NaN
                valid_std = ~grouped['std'].isna()
                if valid_std.any():
                    upper_bounds = grouped['mean'] + grouped['std']
                    all_upper_bounds.extend(upper_bounds[valid_std].tolist())
                    ax.errorbar(grouped.index[valid_std], grouped['mean'][valid_std], 
                               yerr=grouped['std'][valid_std],
                               fmt='none', ecolor=PlottingConfig.COLORS[method],
                               elinewidth=1, capsize=3)
        
        _setup_axis_scaling(ax, log_scale, all_means, all_upper_bounds if show_std else None, global_ylim)
        
        ax.set_xlabel('Sigma (Noise Level)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        # Original hardcoded positioning: ax.legend(loc='upper left')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            pdf_path = _ensure_pdf_extension(save_path)
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by sigma: {str(e)}")
        plt.close()
        return None

def plot_metric_by_variance_explained(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                                    show_std: bool = True, log_scale: bool = False,
                                    global_ylim: Optional[Tuple[float, float]] = None) -> Optional[plt.Figure]:
    """Plot metric vs proportion of variance explained with error bars."""
    try:
        df = _load_and_prepare_data(results_path, metric, need_variance=True)
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
                # Only show error bars if std is not NaN
                valid_std = ~grouped['std'].isna()
                if valid_std.any():
                    upper_bounds = grouped['mean'] + grouped['std']
                    all_upper_bounds.extend(upper_bounds[valid_std].tolist())
                    ax.errorbar(grouped.index[valid_std], grouped['mean'][valid_std], 
                               yerr=grouped['std'][valid_std],
                               fmt='none', ecolor=PlottingConfig.COLORS[method],
                               elinewidth=1, capsize=3)
        
        _setup_axis_scaling(ax, log_scale, all_means, all_upper_bounds if show_std else None, global_ylim)
        format_x_axis_decimal(ax)
        
        ax.set_xlabel('Proportion of Variance Explained (PVE)')
        ax.set_ylabel(PlottingConfig.get_metric_label(metric))
        # Original hardcoded positioning: ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rte', 'rie'] else 'upper left')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            pdf_path = _ensure_pdf_extension(save_path)
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error plotting metric by variance explained: {str(e)}")
        plt.close()
        return None





def plot_mse_vs_df_by_k(results_path: Path, target_sigma: float, save_path: Optional[Path] = None,
                       show_std: bool = False, log_scale_mse: bool = False, log_scale_df: bool = False,
                       sigma_tolerance: float = 1e-6, method_filter: Optional[callable] = None,
                       labels: Optional[Dict[str, List[int]]] = None) -> Optional[plt.Figure]:
    """Plot MSE vs DF for each method's k progression at target sigma.
    
    Args:
        labels: Optional dict mapping method names to lists of k values to label.
                Method names should be display names like 'RGS', 'GS'. 
                Example: {'RGS': [1, 20], 'GS': [1, 10, 20]}
                If None, defaults to labeling k=1, 10, 20 for all methods.
    """
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
        
        # Only select numeric columns for aggregation to avoid dtype errors
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns]
        df_avg = df_numeric.groupby('sigma').mean().reset_index()
        available_methods = []
        
        # Explicitly filter to only original_gs/gs and rgs methods for MSE vs DF by k plots
        target_methods = ['original_gs', 'gs', 'rgs']
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
        
        # Set up labels mapping
        if labels is None:
            # Default: label k=1, 10, 20 for all methods
            default_labels = [1, 10, 20]
            labels_to_use = {}
            for method in available_methods:
                method_display_name = PlottingConfig.get_method_label(method)
                labels_to_use[method] = default_labels
        else:
            # Convert display names to internal method names
            labels_to_use = {}
            for display_name, k_list in labels.items():
                # Map display names to internal method names
                for method in available_methods:
                    if PlottingConfig.get_method_label(method) == display_name:
                        labels_to_use[method] = k_list
                        break
            
            # For methods not specified in labels, use empty list (no labels)
            for method in available_methods:
                if method not in labels_to_use:
                    labels_to_use[method] = []
        
        fig, ax = create_figure()
        setup_plot_style()
        
        method_markers = {'lasso': 'o', 'elastic': 's', 'bagged_gs': 'v', 
                         'smeared_gs': '^', 'original_gs': 'D', 'rgs': '*'}
        
        # First pass: collect all data points for smart positioning
        all_method_data = {}
        for method in available_methods:
            mse_cols = [col for col in df_avg.columns if col.startswith(f'mse_by_k_{method}_')]
            k_values = sorted([int(col.split('_')[-1]) for col in mse_cols if int(col.split('_')[-1]) > 0])
            
            mse_values, df_values, valid_k_values = [], [], []
            for k in k_values:
                mse_col = f'mse_by_k_{method}_{k}'
                df_col = f'df_by_k_{method}_{k}'
                
                if mse_col in df_avg.columns and df_col in df_avg.columns:
                    mse_val = df_avg[mse_col].values[0]
                    df_val = df_avg[df_col].values[0]
                    
                    if not np.isnan(mse_val) and not np.isnan(df_val):
                        mse_values.append(mse_val)
                        df_values.append(df_val)
                        valid_k_values.append(k)
            
            if mse_values and df_values:
                all_method_data[method] = {
                    'mse_values': mse_values,
                    'df_values': df_values,
                    'k_values': valid_k_values
                }

        # Second pass: plot and add smart labels
        for method in available_methods:
            if method not in all_method_data:
                continue
                
            data = all_method_data[method]
            mse_values = data['mse_values']
            df_values = data['df_values']
            k_values = data['k_values']
            
            marker = method_markers.get(method, 'o')
            ax.plot(mse_values, df_values, '-', color=PlottingConfig.COLORS[method],
                   linewidth=1.5, label=PlottingConfig.get_method_label(method))
            ax.scatter(mse_values, df_values, color=PlottingConfig.COLORS[method],
                      marker=marker, s=60, zorder=3, edgecolors='black', linewidths=0.5)
            
            # Add k labels with smart positioning
            for i, k in enumerate(k_values):
                # Check if this k value should be labeled for this method
                if k not in labels_to_use.get(method, []):
                    continue
                
                label = f'k={k}'
                
                current_point = (mse_values[i], df_values[i])
                
                # Calculate global data ranges for crowding detection
                all_mse_vals = []
                all_df_vals = []
                for method_data in all_method_data.values():
                    all_mse_vals.extend(method_data['mse_values'])
                    all_df_vals.extend(method_data['df_values'])
                
                global_mse_range = max(all_mse_vals) - min(all_mse_vals) if len(all_mse_vals) > 1 else max(all_mse_vals)
                global_df_range = max(all_df_vals) - min(all_df_vals) if len(all_df_vals) > 1 else max(all_df_vals)
                
                # Function to calculate line curvature at a point
                def calculate_curvature(point_idx, mse_vals, df_vals):
                    """Calculate curvature at a point using neighboring points."""
                    if point_idx == 0 or point_idx == len(mse_vals) - 1:
                        return 0  # No curvature at endpoints
                    
                    # Get three consecutive points
                    p1 = (mse_vals[point_idx - 1], df_vals[point_idx - 1])
                    p2 = (mse_vals[point_idx], df_vals[point_idx])
                    p3 = (mse_vals[point_idx + 1], df_vals[point_idx + 1])
                    
                    # Calculate vectors
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    
                    # Normalize by data ranges to make curvature scale-invariant
                    v1_norm = (v1[0] / global_mse_range, v1[1] / global_df_range)
                    v2_norm = (v2[0] / global_mse_range, v2[1] / global_df_range)
                    
                    # Calculate angle change (curvature indicator)
                    dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                    v1_mag = (v1_norm[0]**2 + v1_norm[1]**2)**0.5
                    v2_mag = (v2_norm[0]**2 + v2_norm[1]**2)**0.5
                    
                    if v1_mag == 0 or v2_mag == 0:
                        return 0
                    
                    cos_angle = dot_product / (v1_mag * v2_mag)
                    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                    
                    # Return angle change (higher = more curvature)
                    return abs(np.arccos(cos_angle))
                
                def get_perpendicular_offset(point_idx, mse_vals, df_vals, base_distance=12):
                    """Calculate perpendicular offset based on local line direction."""
                    if len(mse_vals) < 2:
                        return (base_distance, 0)
                    
                    # Calculate local line direction
                    if point_idx == 0:
                        # Use direction to next point
                        dx = mse_vals[1] - mse_vals[0]
                        dy = df_vals[1] - df_vals[0]
                    elif point_idx == len(mse_vals) - 1:
                        # Use direction from previous point
                        dx = mse_vals[-1] - mse_vals[-2]
                        dy = df_vals[-1] - df_vals[-2]
                    else:
                        # Use average direction of neighboring segments
                        dx = (mse_vals[point_idx + 1] - mse_vals[point_idx - 1]) / 2
                        dy = (df_vals[point_idx + 1] - df_vals[point_idx - 1]) / 2
                    
                    # Normalize direction vector
                    length = (dx**2 + dy**2)**0.5
                    if length == 0:
                        return (base_distance, 0)
                    
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Get perpendicular vector (rotate 90 degrees)
                    perp_x = -dy_norm
                    perp_y = dx_norm
                    
                    # Convert to pixel offset
                    return (perp_x * base_distance, perp_y * base_distance)
                
                # Curvature-aware positioning with hierarchical strategy
                if method in ['gs', 'original_gs']:
                    # Default: GS labels to the right
                    default_side = 'right'
                    base_offset = (12, 0)
                    
                    # Special handling for k=1 (keep it simple)
                    if k == 1:
                        label_offset = base_offset
                        label_align = 'left'
                    else:
                        # Step 1: Check curvature
                        curvature = calculate_curvature(i, mse_values, df_values)
                        high_curvature = curvature > 0.3  # Lower threshold for more sensitive detection
                        
                        current_side = default_side
                        
                        # Step 1: If high curvature, find the side with more clear space
                        if high_curvature:
                            # Calculate which side has more clear space
                            if i > 0 and i < len(mse_values) - 1:
                                # Get the three points
                                p1 = (mse_values[i-1], df_values[i-1])
                                p2 = (mse_values[i], df_values[i])
                                p3 = (mse_values[i+1], df_values[i+1])
                                
                                # Calculate directions of incoming and outgoing line segments
                                incoming_dir = (p2[0] - p1[0], p2[1] - p1[1])
                                outgoing_dir = (p3[0] - p2[0], p3[1] - p2[1])
                                
                                # Normalize directions
                                incoming_len = (incoming_dir[0]**2 + incoming_dir[1]**2)**0.5
                                outgoing_len = (outgoing_dir[0]**2 + outgoing_dir[1]**2)**0.5
                                
                                if incoming_len > 0 and outgoing_len > 0:
                                    incoming_norm = (incoming_dir[0]/incoming_len, incoming_dir[1]/incoming_len)
                                    outgoing_norm = (outgoing_dir[0]/outgoing_len, outgoing_dir[1]/outgoing_len)
                                    
                                    # Calculate potential label positions (left and right of point)
                                    left_pos = (p2[0] - 0.05 * global_mse_range, p2[1])
                                    right_pos = (p2[0] + 0.05 * global_mse_range, p2[1])
                                    
                                    # Calculate distance from label positions to line segments
                                    def distance_to_line_segment(point, seg_start, seg_end):
                                        # Distance from point to line segment
                                        seg_vec = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
                                        point_vec = (point[0] - seg_start[0], point[1] - seg_start[1])
                                        
                                        seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2
                                        if seg_len_sq == 0:
                                            return ((point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2)**0.5
                                        
                                        t = max(0, min(1, (point_vec[0]*seg_vec[0] + point_vec[1]*seg_vec[1]) / seg_len_sq))
                                        projection = (seg_start[0] + t*seg_vec[0], seg_start[1] + t*seg_vec[1])
                                        
                                        return ((point[0] - projection[0])**2 + (point[1] - projection[1])**2)**0.5
                                    
                                    # Calculate clearance for left and right positions against ALL line segments
                                    left_clearances = []
                                    right_clearances = []
                                    
                                    # Check against all line segments in the curve
                                    for j in range(len(mse_values) - 1):
                                        seg_start = (mse_values[j], df_values[j])
                                        seg_end = (mse_values[j + 1], df_values[j + 1])
                                        
                                        left_clearances.append(distance_to_line_segment(left_pos, seg_start, seg_end))
                                        right_clearances.append(distance_to_line_segment(right_pos, seg_start, seg_end))
                                    
                                    # Use the minimum clearance (closest approach to any segment)
                                    left_clearance = min(left_clearances) if left_clearances else 0
                                    right_clearance = min(right_clearances) if right_clearances else 0
                                    
                                    # Choose the side with more clearance
                                    if right_clearance > left_clearance:
                                        current_side = 'right'
                                    else:
                                        current_side = 'left'
                                else:
                                    # Fallback if we can't calculate directions
                                    current_side = 'left'  # Swap from default
                            else:
                                current_side = 'left'  # Default swap for edge cases
                        
                        # Step 2: Use adaptive positioning based on line direction
                        if high_curvature:
                            perp_offset = get_perpendicular_offset(i, mse_values, df_values, 12)
                            if current_side == 'left':
                                label_offset = (-abs(perp_offset[0]), perp_offset[1])
                                label_align = 'right'
                            else:
                                label_offset = (abs(perp_offset[0]), perp_offset[1])
                                label_align = 'left'
                        else:
                            # Normal positioning
                            if current_side == 'left':
                                label_offset = (-12, 0)
                                label_align = 'right'
                            else:
                                label_offset = (12, 0)
                                label_align = 'left'
                        
                        # Step 3: If still very high curvature, move further away
                        if curvature > 1.0:  # Very high curvature threshold
                            label_offset = (label_offset[0] * 1.5, label_offset[1] * 1.5)
                        
                elif method == 'rgs':
                    # Default: RGS labels to the left
                    default_side = 'left'
                    
                    # Special handling for k=1 (keep it lower as requested)
                    if k == 1:
                        label_offset = (-12, -8)
                        label_align = 'right'
                    else:
                        # Step 1: Check curvature
                        curvature = calculate_curvature(i, mse_values, df_values)
                        high_curvature = curvature > 0.3  # Lower threshold for more sensitive detection
                        
                        current_side = default_side
                        
                        # Step 1: If high curvature, find the side with more clear space
                        if high_curvature:
                            # Calculate which side has more clear space
                            if i > 0 and i < len(mse_values) - 1:
                                # Get the three points
                                p1 = (mse_values[i-1], df_values[i-1])
                                p2 = (mse_values[i], df_values[i])
                                p3 = (mse_values[i+1], df_values[i+1])
                                
                                # Calculate directions of incoming and outgoing line segments
                                incoming_dir = (p2[0] - p1[0], p2[1] - p1[1])
                                outgoing_dir = (p3[0] - p2[0], p3[1] - p2[1])
                                
                                # Normalize directions
                                incoming_len = (incoming_dir[0]**2 + incoming_dir[1]**2)**0.5
                                outgoing_len = (outgoing_dir[0]**2 + outgoing_dir[1]**2)**0.5
                                
                                if incoming_len > 0 and outgoing_len > 0:
                                    incoming_norm = (incoming_dir[0]/incoming_len, incoming_dir[1]/incoming_len)
                                    outgoing_norm = (outgoing_dir[0]/outgoing_len, outgoing_dir[1]/outgoing_len)
                                    
                                    # Calculate potential label positions (left and right of point)
                                    left_pos = (p2[0] - 0.05 * global_mse_range, p2[1])
                                    right_pos = (p2[0] + 0.05 * global_mse_range, p2[1])
                                    
                                    # Calculate distance from label positions to line segments
                                    def distance_to_line_segment(point, seg_start, seg_end):
                                        # Distance from point to line segment
                                        seg_vec = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
                                        point_vec = (point[0] - seg_start[0], point[1] - seg_start[1])
                                        
                                        seg_len_sq = seg_vec[0]**2 + seg_vec[1]**2
                                        if seg_len_sq == 0:
                                            return ((point[0] - seg_start[0])**2 + (point[1] - seg_start[1])**2)**0.5
                                        
                                        t = max(0, min(1, (point_vec[0]*seg_vec[0] + point_vec[1]*seg_vec[1]) / seg_len_sq))
                                        projection = (seg_start[0] + t*seg_vec[0], seg_start[1] + t*seg_vec[1])
                                        
                                        return ((point[0] - projection[0])**2 + (point[1] - projection[1])**2)**0.5
                                    
                                    # Calculate clearance for left and right positions against ALL line segments
                                    left_clearances = []
                                    right_clearances = []
                                    
                                    # Check against all line segments in the curve
                                    for j in range(len(mse_values) - 1):
                                        seg_start = (mse_values[j], df_values[j])
                                        seg_end = (mse_values[j + 1], df_values[j + 1])
                                        
                                        left_clearances.append(distance_to_line_segment(left_pos, seg_start, seg_end))
                                        right_clearances.append(distance_to_line_segment(right_pos, seg_start, seg_end))
                                    
                                    # Use the minimum clearance (closest approach to any segment)
                                    left_clearance = min(left_clearances) if left_clearances else 0
                                    right_clearance = min(right_clearances) if right_clearances else 0
                                    
                                    # Choose the side with more clearance
                                    if right_clearance > left_clearance:
                                        current_side = 'right'
                                    else:
                                        current_side = 'left'
                                else:
                                    # Fallback if we can't calculate directions
                                    current_side = 'right'  # Swap from default
                            else:
                                current_side = 'right'  # Default swap for edge cases
                        
                        # Step 2: Use adaptive positioning based on line direction
                        if high_curvature:
                            perp_offset = get_perpendicular_offset(i, mse_values, df_values, 12)
                            if current_side == 'right':
                                label_offset = (abs(perp_offset[0]), perp_offset[1])
                                label_align = 'left'
                            else:
                                label_offset = (-abs(perp_offset[0]), perp_offset[1])
                                label_align = 'right'
                        else:
                            # Normal positioning
                            if current_side == 'right':
                                label_offset = (12, 0)
                                label_align = 'left'
                            else:
                                label_offset = (-12, 0)
                                label_align = 'right'
                        
                        # Step 3: If still very high curvature, move further away
                        if curvature > 1.0:
                            label_offset = (label_offset[0] * 1.5, label_offset[1] * 1.5)
                        
                else:
                    # Default positioning for other methods
                    label_offset = (0, 8)
                    label_align = 'center'
                
                ax.annotate(
                    label,
                    current_point,
                    textcoords="offset points",
                    xytext=label_offset,
                    ha=label_align,
                    fontsize=10,
                    fontweight='bold',
                    bbox=None
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
            # Set x-axis limits with extra padding to ensure labels are visible
            if all_mse_values:
                x_min, x_max = min(all_mse_values), max(all_mse_values)
                x_range = x_max - x_min
                # Add extra padding: 12% on left for RGS labels, 15% on right for GS labels
                left_padding = x_range * 0.12 if x_range > 0 else x_max * 0.12
                right_padding = x_range * 0.15 if x_range > 0 else x_max * 0.15
                ax.set_xlim(left=x_min - left_padding, right=x_max + right_padding)
        
        if log_scale_df:
            ax.set_yscale('log')
        else:
            # Set y-axis limits with padding to ensure markers are visible  
            if all_df_values:
                y_min, y_max = min(all_df_values), max(all_df_values)
                y_range = y_max - y_min
                y_padding = y_range * 0.08 if y_range > 0 else y_max * 0.08
                ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
        
        ax.set_xlabel('Training MSE')
        ax.set_ylabel('Degrees of Freedom (DF)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            pdf_path = _ensure_pdf_extension(save_path)
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error plotting MSE vs DF: {str(e)}")
        plt.close()
        return None

def barplot_metric_by_sigma(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                           show_std: bool = True, log_scale: bool = False,
                           global_ylim: Optional[Tuple[float, float]] = None) -> Optional[plt.Figure]:
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
                # Only show error bars where std is not NaN
                valid_indices = [i for i, std in enumerate(stds) if not np.isnan(std)]
                if valid_indices:
                    valid_positions = [positions[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_stds = [stds[i] for i in valid_indices]
                    ax.errorbar(valid_positions, valid_means, yerr=valid_stds, fmt='none',
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
        # Original hardcoded positioning: ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rie'] else 'upper left')
        ax.legend(loc='best')
        ax.grid(True, axis='y', alpha=0.3)

        # Use global limits if provided, otherwise calculate from data
        if global_ylim:
            ax.set_ylim(bottom=global_ylim[0], top=global_ylim[1])
        else:
            # Calculate appropriate y-limits from the data, including error bars if shown
            all_values = []
            for method in sorted_methods:
                metric_col = f'{metric}_{method}'
                for snr in snr_values:
                    snr_data = df[df['snr'] == snr]
                    mean = snr_data[metric_col].mean()
                    if show_std:
                        std = snr_data[metric_col].std()
                        if not np.isnan(std):
                            all_values.extend([mean - std, mean + std])
                        else:
                            all_values.append(mean)
                    else:
                        all_values.append(mean)

            if all_values:
                min_val = min(all_values)
                max_val = max(all_values) 
                data_range = max_val - min_val
                padding = data_range * 0.05
                ax.set_ylim(bottom=min_val - padding, top=max_val + padding)
        
        if save_path:
            pdf_path = _ensure_pdf_extension(save_path)
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig
        
    except Exception as e:
        print(f"Error creating bar plot by sigma: {str(e)}")
        plt.close()
        return None

def barplot_metric_by_variance_explained(results_path: Path, metric: str = 'mse', save_path: Optional[Path] = None,
                                       show_std: bool = True, log_scale: bool = True,
                                       global_ylim: Optional[Tuple[float, float]] = None) -> Optional[plt.Figure]:
    """Create bar plot of metric by variance explained."""
    try:
        df = _load_and_prepare_data(results_path, metric, need_variance=True)
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
                # Only show error bars where std is not NaN
                valid_indices = [i for i, std in enumerate(stds) if not np.isnan(std)]
                if valid_indices:
                    valid_positions = [positions[i] for i in valid_indices]
                    valid_means = [means[i] for i in valid_indices]
                    valid_stds = [stds[i] for i in valid_indices]
                    ax.errorbar(valid_positions, valid_means, yerr=valid_stds, fmt='none',
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
        # Original hardcoded positioning: ax.legend(loc='upper right' if metric in ['mse', 'insample', 'outsample_mse', 'rie'] else 'upper left')
        ax.legend(loc='best')
        ax.grid(True, axis='y', alpha=0.3)

        # Use global limits if provided, otherwise calculate from data
        if global_ylim:
            ax.set_ylim(bottom=global_ylim[0], top=global_ylim[1])
        elif not log_scale:
            # Calculate appropriate y-limits for ALL metrics, including error bars if shown
            all_values = []
            for method in available_methods:
                metric_col = f'{metric}_{method}'
                for var_expl in var_explained_values:
                    var_data = df[np.isclose(df['var_explained_rounded'], var_expl)]
                    mean = var_data[metric_col].mean()
                    if show_std:
                        std = var_data[metric_col].std()
                        if not np.isnan(std):
                            all_values.extend([mean - std, mean + std])
                        else:
                            all_values.append(mean)
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
            pdf_path = _ensure_pdf_extension(save_path)
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
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

def collect_global_y_limits(results_files: List[Path], metric: str, show_std: bool = True, 
                           log_scale: bool = False) -> Optional[Tuple[float, float]]:
    """Collect global y-limits across multiple result files for consistent scaling."""
    all_values = []
    
    for results_path in results_files:
        try:
            # Load data based on metric type
            if metric in ['rie']:
                df = _load_and_prepare_data(results_path, metric)
            elif metric in ['mse', 'outsample_mse', 'coef_recovery', 'rte']:
                df = _load_and_prepare_data(results_path, metric, need_variance=True)
            else:
                df = _load_and_prepare_data(results_path, metric)
            
            available_methods = get_available_methods(df, metric)
            if not available_methods:
                continue
                
            # Collect values for each method
            for method in available_methods:
                metric_col = f'{metric}_{method}'
                if metric_col not in df.columns or df[metric_col].isna().all():
                    continue
                    
                # Get values and add to collection
                values = df[metric_col].dropna()
                all_values.extend(values.tolist())
                
                # If showing std, also include bounds with error bars
                if show_std:
                    # Group by appropriate x-axis and calculate std
                    if 'var_explained' in df.columns:
                        grouped = df.groupby('var_explained')[metric_col].agg(['mean', 'std'])
                    else:
                        grouped = df.groupby('sigma')[metric_col].agg(['mean', 'std'])
                    
                    # Add error bar bounds only if std is not NaN
                    valid_std = ~grouped['std'].isna()
                    if valid_std.any():
                        upper_bounds = grouped['mean'] + grouped['std']
                        lower_bounds = grouped['mean'] - grouped['std']
                        all_values.extend(upper_bounds[valid_std].tolist())
                        all_values.extend(lower_bounds[valid_std].tolist())
                    
        except Exception as e:
            print(f"Warning: Could not process {results_path.name} for global scaling: {str(e)}")
            continue
    
    if not all_values:
        return None
        
    # Remove any nan or inf values
    all_values = [v for v in all_values if np.isfinite(v)]
    if not all_values:
        return None
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Add padding
    if log_scale:
        # For log scale, use multiplicative padding
        log_min = np.log10(max(min_val, 1e-10))  # Avoid log(0)
        log_max = np.log10(max_val)
        log_range = log_max - log_min
        padding = log_range * 0.05
        y_min = 10 ** (log_min - padding)
        y_max = 10 ** (log_max + padding)
    else:
        # For linear scale, use additive padding
        data_range = max_val - min_val
        padding = data_range * 0.05 if data_range > 0 else max_val * 0.05
        y_min = max(0, min_val - padding) if min_val >= 0 else min_val - padding
        y_max = max_val + padding
    
    return (y_min, y_max) 