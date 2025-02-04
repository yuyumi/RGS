import sys
from pathlib import Path
import json
from glob import glob
from datetime import datetime
import argparse

from simulation_main import main as run_simulation


def run_multiple_simulations(params_dir='params', pattern=None):
    # Get parameter files
    params_path = Path(params_dir)
    if pattern:
        param_files = list(params_path.glob(f'*{pattern}*.json'))
    else:
        param_files = list(params_path.glob('*.json'))
    
    if not param_files:
        print(f"No parameter files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(param_files)} parameter files:")
    for i, param_file in enumerate(param_files, 1):
        print(f"{i}. {param_file.name}")
    
    # Run each simulation
    for param_file in param_files:
        print(f"\n{'='*80}")
        print(f"Running simulation with parameters from: {param_file.name}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*80)
        
        try:
            results_df, summary = run_simulation(param_file)
            print(f"Simulation completed successfully: {param_file.name}")
        except Exception as e:
            print(f"Error running simulation {param_file.name}:")
            print(f"Error message: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple RGS simulations')
    parser.add_argument('--pattern', type=str, help='Pattern to match parameter files (e.g., "banded" for all banded simulations)')
    parser.add_argument('--params-dir', type=str, default='params', help='Directory containing parameter files')
    
    args = parser.parse_args()
    run_multiple_simulations(args.params_dir, args.pattern)