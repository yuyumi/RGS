from pathlib import Path
import json
import argparse
from typing import Optional, List, Dict, Tuple

from rgs_experiments.plotting.plotting import *

def extract_metric_from_plot_name(plot_name: str) -> str:
    """Extract the metric name from plot name for global scaling lookup."""
    # Handle special cases first
    if plot_name.startswith('coef_recovery'):
        return 'coef_recovery'
    elif plot_name.startswith('outsample'):
        return 'outsample_mse'
    elif plot_name.startswith('insample'):
        return 'insample'
    elif plot_name.startswith('mse'):
        return 'mse'
    elif plot_name.startswith('rte'):
        return 'rte'
    elif plot_name.startswith('rie'):
        return 'rie'
    else:
        # Fallback to first part
        return plot_name.split('_')[0]

def create_plots_for_result(
    results_path: Path,
    figures_dir: Path,
    params_path: Optional[Path] = None,
    show_std: bool = False,
    plot_type: str = 'both',  # Parameter to choose between 'line', 'bar', or 'both'
    global_ylim_dict: Optional[Dict[str, Tuple[float, float]]] = None
) -> None:
    """Create all plots for a single simulation result file."""
    print(f"\nProcessing: {results_path.name} with {plot_type} plots")
    
    # Base name for plot files
    base_name = results_path.stem.replace('simulation_results_', '')
    
    # Read params if available to get reference sigma values
    sigma_values = None
    if params_path and params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
            if 'simulation' in params and 'sigma' in params['simulation']:
                if 'computed_values' in params['simulation']['sigma']:
                    sigma_values = params['simulation']['sigma']['computed_values']
    
    # If no params file, use unique sigma values from results
    if sigma_values is None:
        import pandas as pd
        df = pd.read_csv(results_path)
        sigma_values = sorted(df['sigma'].unique())
    
    # Generate MSE vs DF plot (only for line plots)
    if plot_type == 'line' or plot_type == 'both':
        try:
            print("Attempting to create MSE vs DF by k plots...")
            
            for sigma in sigma_values:
                save_path = figures_dir / f"mse_vs_df_by_k_sigma_{sigma:.3f}_{base_name}.png"
                
                # Call the function with the method filter
                plot_mse_vs_df_by_k(
                    results_path=results_path, 
                    target_sigma=sigma, 
                    save_path=save_path, 
                    show_std=show_std, 
                    method_filter=lambda m: m in ['rgs', 'original_gs']
                )
                # Note: MSE vs DF plots don't use global scaling as they have different axis meanings
                print(f"Created: {save_path.name}")
        except Exception as e:
            print(f"Error creating mse_vs_df_by_k plot: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
    
    # Define plot configurations based on plot_type
    if plot_type == 'line' or plot_type == 'both':
        plot_types = {
            # 'mse_sigma': (plot_mse_by_sigma, {}),
            # 'df_sigma': (plot_df_by_sigma, {}),
            # 'insample_sigma': (plot_insample_by_sigma, {}),
            # 'outsample_sigma': (plot_outsample_mse_by_sigma, {}),
            'mse_pve': (plot_mse_by_variance_explained, {}),
            # 'insample_pve': (plot_insample_by_variance_explained, {}),
            'outsample_pve': (plot_outsample_mse_by_variance_explained, {}),
            'coef_recovery_pve': (plot_coef_recovery_by_variance_explained, {}),
            'rte_pve': (plot_rte_by_variance_explained, {})
        }
        
        # Generate basic plots
        for plot_name, (plot_func, plot_kwargs) in plot_types.items():
            try:
                # Special case for df plots - exclude Ridge
                if 'df' in plot_name:
                    # Create a filtering function
                    def filtered_plot_func(*args, **kwargs):
                        # Add method_filter to kwargs
                        kwargs['method_filter'] = lambda m: m in ['rgs', 'original_gs']
                        return plot_func(*args, **kwargs)
                    
                    save_path = figures_dir / f"{plot_name}_{base_name}.png"
                    filtered_plot_func(results_path, save_path=save_path, show_std=show_std, **plot_kwargs)
                else:
                    save_path = figures_dir / f"{plot_name}_{base_name}.png"
                    # Add global y-limits if available for this metric
                    metric = extract_metric_from_plot_name(plot_name)
                    if global_ylim_dict and metric in global_ylim_dict:
                        plot_kwargs['global_ylim'] = global_ylim_dict[metric]
                    plot_func(results_path, save_path=save_path, show_std=show_std, **plot_kwargs)
                
                print(f"Created: {save_path.name}")
            except Exception as e:
                print(f"Error creating {plot_name}: {str(e)}")
    
    if plot_type == 'bar' or plot_type == 'both':
        plot_types = {
            # 'mse_sigma': (barplot_mse_by_sigma, {'log_scale': True}),
            # 'df_sigma': (barplot_df_by_sigma, {'log_scale': False}),
            # 'insample_sigma': (barplot_insample_by_sigma, {'log_scale': True}),
            # 'outsample_sigma': (barplot_outsample_mse_by_sigma, {'log_scale': True}),
            'mse_pve': (barplot_mse_by_variance_explained, {'log_scale': True}),
            'insample_pve': (barplot_insample_by_variance_explained, {'log_scale': True}),
            'outsample_pve': (barplot_outsample_mse_by_variance_explained, {'log_scale': True}),
            'rie_pve': (barplot_rie_by_variance_explained, {}),
            'rte_pve': (barplot_rte_by_variance_explained, {'log_scale': False}),
            'rie_sigma': (barplot_rie_by_sigma, {'log_scale': False}),
            'rte_sigma': (barplot_rte_by_sigma, {'log_scale': False})
        }
        
        # Generate basic plots
        for plot_name, (plot_func, plot_kwargs) in plot_types.items():
            try:
                # Special case for df plots - exclude Ridge
                if 'df' in plot_name:
                    # Create a filtering function
                    def filtered_plot_func(*args, **kwargs):
                        # Add method_filter to kwargs
                        kwargs['method_filter'] = lambda m: m != 'ridge'
                        return plot_func(*args, **kwargs)
                    
                    save_path = figures_dir / f"{plot_name}_bar_{base_name}.png"
                    filtered_plot_func(results_path, save_path=save_path, show_std=show_std, **plot_kwargs)
                else:
                    save_path = figures_dir / f"{plot_name}_bar_{base_name}.png"
                    # Add global y-limits if available for this metric
                    metric = extract_metric_from_plot_name(plot_name)
                    if global_ylim_dict and metric in global_ylim_dict:
                        plot_kwargs['global_ylim'] = global_ylim_dict[metric]
                    plot_func(results_path, save_path=save_path, show_std=show_std, **plot_kwargs)
                
                print(f"Created: {save_path.name}")
            except Exception as e:
                print(f"Error creating {plot_name}: {str(e)}")

def run_plotting(results_dir: Optional[str] = None, pattern: Optional[str] = None, plot_type: str = 'both', show_std: bool = True, global_scale: bool = True) -> None:
    """Run plotting for all simulation results matching the pattern."""
    # Get project root directory (two levels up from this script)
    root_dir = Path(__file__).parent.parent
    
    # Use default results directory if none specified
    if results_dir is None:
        raw_results_path = root_dir / "results" / "raw"
    else:
        raw_results_path = Path(results_dir)
    
    if not raw_results_path.exists():
        print(f"Results directory not found: {raw_results_path}")
        return
    
    # Create figures directory in results folder
    figures_dir = root_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all results files
    pattern_str = f'simulation_results_*{pattern}*.csv' if pattern else 'simulation_results_*.csv'
    results_files = list(raw_results_path.glob(pattern_str))
    
    print(f"Searching in directory: {raw_results_path}")
    print(f"Using pattern: {pattern_str}")
    print(f"Plot type: {plot_type}")
    print(f"Saving figures to: {figures_dir}")
    
    if not results_files:
        print(f"No result files found matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(results_files)} result files:")
    for i, result_file in enumerate(results_files, 1):
        print(f"{i}. {result_file.name}")
    
    # Collect global y-limits if global_scale is enabled
    global_ylim_dict = None
    if global_scale:
        print("\nCollecting global y-limits for consistent scaling...")
        global_ylim_dict = {}
        
        # Define metrics to collect global limits for
        metrics_to_scale = ['mse', 'outsample_mse', 'coef_recovery', 'rte', 'rie', 'insample']
        
        for metric in metrics_to_scale:
            try:
                # Determine log_scale based on plot configuration
                log_scale = False
                if metric in ['mse', 'outsample_mse', 'insample']:
                    log_scale = True  # These typically use log scale in bar plots
                
                global_ylim = collect_global_y_limits(
                    results_files, 
                    metric, 
                    show_std=show_std, 
                    log_scale=log_scale
                )
                if global_ylim:
                    global_ylim_dict[metric] = global_ylim
                    print(f"  {metric}: y-limits = ({global_ylim[0]:.3e}, {global_ylim[1]:.3e})")
            except Exception as e:
                print(f"  Warning: Could not collect global limits for {metric}: {str(e)}")
        
        if not global_ylim_dict:
            print("  No global limits collected, using individual scaling")
            global_ylim_dict = None
        else:
            print(f"  Collected global limits for {len(global_ylim_dict)} metrics")
    
    # Process each results file
    for result_file in results_files:
        try:
            # Convert result_file to Path if it isn't already
            result_file = Path(result_file)
            
            # Generate params file path
            params_name = result_file.name.replace('simulation_results_', 'simulation_params_')
            params_file = result_file.parent / params_name.replace('.csv', '.json')
            
            create_plots_for_result(
                result_file, 
                figures_dir, 
                params_file, 
                show_std=show_std,
                plot_type=plot_type,
                global_ylim_dict=global_ylim_dict
            )
        except Exception as e:
            print(f"Error processing {result_file.name}:")
            print(f"Error message: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots for RGS simulation results')
    parser.add_argument('--pattern', type=str, help='Pattern to match result files')
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory containing raw result files (default: PROJECT_ROOT/results/raw)')
    parser.add_argument('--no-std', action='store_false', dest='show_std',
                  help='Disable standard deviation bands/error bars in plots')
    parser.set_defaults(show_std=True)
    parser.add_argument('--plot-type', type=str, choices=['line', 'bar', 'both'], default='both',
                      help='Type of plots to generate: line, bar, or both (default: both)')
    parser.add_argument('--no-global-scale', action='store_false', dest='global_scale',
                      help='Disable global y-axis scaling (each plot uses its own scale)')
    parser.set_defaults(global_scale=True)
    
    args = parser.parse_args()
    
    run_plotting(args.results_dir, args.pattern, args.plot_type, args.show_std, args.global_scale)