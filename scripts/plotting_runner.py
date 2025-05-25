from pathlib import Path
import json
import argparse
from typing import Optional, List, Dict

from rgs_experiments.plotting.plotting import plot_metric

def create_plots_for_result(
    results_path: Path,
    figures_dir: Path,
    params_path: Optional[Path] = None,
    show_std: bool = False,
    plot_type: str = 'both'
) -> None:
    """Create all plots for a single simulation result file using unified plot_metric function."""
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
    
    # Generate MSE vs DF plots for each sigma value
    if plot_type == 'line' or plot_type == 'both':
        try:
            print("Creating MSE vs DF plots for each sigma...")
            
            for sigma in sigma_values:
                sigma_str = f"{sigma:.3f}".replace('.', '_')
                save_path = figures_dir / f"mse_vs_df_sigma_{sigma_str}_{base_name}.png"
                
                plot_metric(
                    results_path=results_path,
                    metric='mse_vs_df',
                    x_variable='fixed_sigma',
                    plot_type='scatter_line',
                    target_sigma=sigma,
                    log_scale_x=False,
                    log_scale_y=False,
                    method_filter=lambda m: m in ['rgs', 'original_gs'],
                    save_path=save_path
                )
                print(f"Created: {save_path.name}")
        except Exception as e:
            print(f"Error creating mse_vs_df plots: {str(e)}")
    
    # Define all plot configurations
    plot_configs = [
        # Line plots
        {'metric': 'mse', 'x_variable': 'variance', 'plot_type': 'line', 'log_scale': False, 'enabled_for': ['line', 'both']},
        {'metric': 'outsample_mse', 'x_variable': 'variance', 'plot_type': 'line', 'log_scale': False, 'enabled_for': ['line', 'both']},
        {'metric': 'coef_recovery', 'x_variable': 'variance', 'plot_type': 'line', 'log_scale': False, 'enabled_for': ['line', 'both']},
        {'metric': 'rte', 'x_variable': 'variance', 'plot_type': 'line', 'log_scale': False, 'enabled_for': ['line', 'both']},
        
        # Bar plots
        {'metric': 'mse', 'x_variable': 'variance', 'plot_type': 'bar', 'log_scale': True, 'enabled_for': ['bar', 'both']},
        {'metric': 'insample', 'x_variable': 'variance', 'plot_type': 'bar', 'log_scale': True, 'enabled_for': ['bar', 'both']},
        {'metric': 'outsample_mse', 'x_variable': 'variance', 'plot_type': 'bar', 'log_scale': True, 'enabled_for': ['bar', 'both']},
        {'metric': 'rie', 'x_variable': 'variance', 'plot_type': 'bar', 'log_scale': False, 'enabled_for': ['bar', 'both']},
        {'metric': 'rte', 'x_variable': 'variance', 'plot_type': 'bar', 'log_scale': False, 'enabled_for': ['bar', 'both']},
    ]
    
    # Generate plots based on configurations
    for config in plot_configs:
        if plot_type not in config['enabled_for']:
            continue
            
        try:
            # Generate filename
            x_suffix = 'pve' if config['x_variable'] == 'variance' else 'sigma'
            plot_suffix = f"_{config['plot_type']}" if config['plot_type'] == 'bar' else ""
            filename = f"{config['metric']}_{x_suffix}{plot_suffix}_{base_name}.png"
            save_path = figures_dir / filename
            
            # Determine method filter for DF-related plots
            method_filter = None
            if 'df' in config['metric']:
                method_filter = lambda m: m in ['rgs', 'original_gs']
            
            # Create the plot
            plot_metric(
                results_path=results_path,
                metric=config['metric'],
                x_variable=config['x_variable'],
                plot_type=config['plot_type'],
                show_std=show_std,
                log_scale=config['log_scale'],
                method_filter=method_filter,
                save_path=save_path
            )
            
            print(f"Created: {save_path.name}")
            
        except Exception as e:
            print(f"Error creating {config['metric']} {config['plot_type']} plot: {str(e)}")


def run_plotting(results_dir: Optional[str] = None, pattern: Optional[str] = None, plot_type: str = 'both', show_std: bool = True) -> None:
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
                plot_type=plot_type
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
    
    args = parser.parse_args()
    
    run_plotting(args.results_dir, args.pattern, args.plot_type, args.show_std)