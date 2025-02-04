# scripts/plotting_runner.py
from pathlib import Path
import json
import argparse

from rgs_experiments.plotting.plotting import (
    plot_mse_by_sigma,
    plot_mse_by_variance_explained,
    plot_df_by_sigma,
    plot_df_by_variance_explained,
    plot_mse_vs_k,
    plot_df_vs_k
)

def create_plots_for_result(results_path, figures_dir, params_path=None):
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
    
    # Create plots
    plot_types = {
        'mse_sigma': plot_mse_by_sigma,
        'mse_pve': plot_mse_by_variance_explained,
        'df_sigma': plot_df_by_sigma,
        'df_pve': plot_df_by_variance_explained
    }
    
    # Generate basic plots
    for plot_name, plot_func in plot_types.items():
        try:
            save_path = figures_dir / f"{plot_name}_{base_name}.png"
            plot_func(results_path, save_path)
            print(f"Created: {save_path.name}")
        except Exception as e:
            print(f"Error creating {plot_name}: {str(e)}")
    
    # Generate k-specific plots for select sigma values
    # Choose a few representative sigma values (e.g., low, medium, high noise)
    k_plot_sigmas = [sigma_values[0], sigma_values[len(sigma_values)//2], sigma_values[-1]]
    
    for sigma in k_plot_sigmas:
        try:
            # MSE vs k plot
            save_path = figures_dir / f"mse_vs_k_sigma_{sigma}_{base_name}.png"
            plot_mse_vs_k(results_path, sigma, save_path)
            print(f"Created: {save_path.name}")
            
            # DF vs k plot
            save_path = figures_dir / f"df_vs_k_sigma_{sigma}_{base_name}.png"
            plot_df_vs_k(results_path, sigma, save_path)
            print(f"Created: {save_path.name}")
        except Exception as e:
            print(f"Error creating k plots for sigma={sigma}: {str(e)}")

def run_plotting(results_dir=None, pattern=None):
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
    if pattern:
        results_files = list(raw_results_path.glob(f'simulation_results_*{pattern}*.csv'))
    else:
        results_files = list(raw_results_path.glob('simulation_results_*.csv'))
    
    print(f"Searching in directory: {raw_results_path}")
    print(f"Using pattern: {'simulation_results_*' + (pattern + '*' if pattern else '') + '.csv'}")
    print(f"Saving figures to: {figures_dir}")
    
    if not results_files:
        print(f"No result files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(results_files)} result files:")
    for i, result_file in enumerate(results_files, 1):
        print(f"{i}. {result_file.name}")
    
    # Process each results file
    for result_file in results_files:
        try:
            # Look for corresponding params file
            params_file = result_file.parent / result_file.name.replace('simulation_results_', 'simulation_params_')
            params_file = params_file.with_suffix('.json')
            
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
    
    args = parser.parse_args()
    run_plotting(args.results_dir, args.pattern)