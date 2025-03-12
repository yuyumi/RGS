from pathlib import Path
import json
import argparse
from typing import Optional, List, Dict

from rgs_experiments.plotting.plotting import *

def create_plots_for_result(
    results_path: Path,
    figures_dir: Path,
    params_path: Optional[Path] = None,
    show_std: bool = False
) -> None:
    """Create all plots for a single simulation result file."""
    print(f"\nProcessing: {results_path.name}")
    
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
    
    # Define plot configurations
    plot_types = {
        # 'mse_sigma': (plot_mse_by_sigma, {}),
        # 'df_sigma': (plot_df_by_sigma, {}),
        # 'insample_sigma': (plot_insample_by_sigma, {}),
        # 'outsample_sigma': (plot_outsample_mse_by_sigma, {}),
        # 'mse_pve': (plot_mse_by_variance_explained, {}),
        # 'df_pve': (plot_df_by_variance_explained, {}),
        'insample_pve': (plot_insample_by_variance_explained, {}),
        'outsample_pve': (plot_outsample_mse_by_variance_explained, {})
    }
    
    # Generate basic plots
    for plot_name, (plot_func, plot_kwargs) in plot_types.items():
        try:
            save_path = figures_dir / f"{plot_name}_{base_name}.png"
            plot_func(str(results_path), save_path=save_path, show_std=show_std, **plot_kwargs)
            print(f"Created: {save_path.name}")
        except Exception as e:
            print(f"Error creating {plot_name}: {str(e)}")
    
    # Generate k-specific plots for select sigma values
    # Choose a few representative sigma values (e.g., low, medium, high noise)
    k_plot_sigmas = [sigma_values[0], sigma_values[len(sigma_values)//2], sigma_values[-1]]
    
    k_plot_types = {
        # 'mse_vs_k': plot_mse_vs_k,
        # 'df_vs_k': plot_df_vs_k,
        # 'insample_vs_k': plot_insample_vs_k
    }
    
    for sigma in k_plot_sigmas:
        for plot_name, plot_func in k_plot_types.items():
            try:
                save_path = figures_dir / f"{plot_name}_sigma_{sigma:.3f}_{base_name}.png"
                plot_func(str(results_path), sigma, save_path=save_path)
                print(f"Created: {save_path.name}")
            except Exception as e:
                print(f"Error creating {plot_name} for sigma={sigma}: {str(e)}")

def run_plotting(results_dir: Optional[str] = None, pattern: Optional[str] = None) -> None:
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
            
            create_plots_for_result(result_file, figures_dir, params_file)
        except Exception as e:
            print(f"Error processing {result_file.name}:")
            print(f"Error message: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots for RGS simulation results')
    parser.add_argument('--pattern', type=str, help='Pattern to match result files')
    parser.add_argument('--results-dir', type=str, default=None,
                      help='Directory containing raw result files (default: PROJECT_ROOT/results/raw)')
    parser.add_argument('--show-std', action='store_true',
                      help='Enable standard deviation bands in plots')
    
    args = parser.parse_args()
    
    show_std = args.show_std
    run_plotting(args.results_dir, args.pattern)