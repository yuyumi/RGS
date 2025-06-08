#!/usr/bin/env python3
"""
Main runner for RFS simulations.

Usage:
    python run_simulation.py                  # Run all param files in params/ folder
    python run_simulation.py [param_file]     # Run specific param file
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from glob import glob

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from simulation.pipeline import SimulationPipeline


def run_multiple_simulations(params_dir='params', pattern=None):
    """
    Run simulations for all parameter files in the specified directory.
    
    Parameters
    ----------
    params_dir : str
        Directory containing parameter files
    pattern : str, optional
        Pattern to match in filenames
    """
    # Get project root and params directory
    project_root = Path(__file__).parent.parent
    params_path = project_root / params_dir
    
    # Check if params directory exists
    if not params_path.exists():
        print(f"Error: Parameter directory not found: {params_path}")
        print(f"Available directories in project root:")
        for d in project_root.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                print(f"   - {d.name}/")
        print(f"\nUse: python run_simulation.py --help")
        return False
    
    if pattern:
        param_files = list(params_path.glob(f'*{pattern}*.json'))
    else:
        param_files = list(params_path.glob('*.json'))
    
    if not param_files:
        print(f"No parameter files found in {params_path}")
        if pattern:
            print(f"Pattern searched: *{pattern}*.json")
        print(f"Available .json files in {params_dir}/:")
        all_json_files = list(params_path.glob('*.json'))
        if all_json_files:
            for f in all_json_files:
                print(f"   - {f.name}")
        else:
            print("   (none)")
        print(f"\nUse: python run_simulation.py --help")
        return False
    
    print(f"Found {len(param_files)} parameter files in {params_dir}/:")
    for i, param_file in enumerate(param_files, 1):
        print(f"{i:2d}. {param_file.name}")
    
    # Run each simulation
    for param_file in param_files:
        print(f"\n{'='*80}")
        print(f"Running simulation with parameters from: {param_file.name}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*80)
        
        try:
            results_df, summary_df, timing_summary_df = SimulationPipeline.run_from_config(str(param_file))
            print(f"Simulation completed successfully: {param_file.name}")
            
        except Exception as e:
            print(f"Error running simulation {param_file.name}:")
            print(f"Error message: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            continue
    
    return True

def run_single_simulation(param_file, validate_only=False):
    """Run a single simulation from a parameter file."""
    # Convert to absolute path if needed
    param_path = Path(param_file)
    if not param_path.is_absolute():
        # Look for the file relative to the project root
        project_root = Path(__file__).parent.parent
        param_path = project_root / param_file
    
    # Check that parameter file exists
    if not param_path.exists():
        print(f"Error: Parameter file not found: {param_path}")
        print(f"Searched in: {param_path.parent}")
        
        # Show available JSON files in the directory
        parent_dir = param_path.parent
        if parent_dir.exists():
            json_files = list(parent_dir.glob('*.json'))
            if json_files:
                print(f"Available .json files in {parent_dir.name}/:")
                for f in json_files:
                    print(f"   - {f.name}")
            else:
                print(f"No .json files found in {parent_dir.name}/")
        
        print(f"\nUse: python run_simulation.py --help")
        sys.exit(1)
    
    print(f"Using parameter file: {param_path}")
    
    try:
        # Initialize pipeline
        pipeline = SimulationPipeline(str(param_path))
        
        if validate_only:
            print("Parameter validation successful")
            print(f"Configuration summary:")
            print(f"  - Simulations: {pipeline.params['simulation']['n_sim']}")
            print(f"  - Data type: {pipeline.params['data']['generator_type']}")
            print(f"  - Covariance: {pipeline.params['data']['covariance_type']}")
            print(f"  - Dimensions: {pipeline.params['data']['n_train']} × {pipeline.params['data']['n_predictors']}")
            return
        
        # Run full simulation
        print(f"Starting simulation with configuration: {param_path.name}")
        results_df, summary_df, timing_summary_df = pipeline.run_full_simulation()
        
        print("\nSimulation completed successfully!")
        print(f"Results available in: {pipeline.params['output']['save_path']}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for RFS simulations."""
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run RFS simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default behavior - run ALL parameter files in params/ folder
    python run_simulation.py                          

    # Run specific parameter file
    python run_simulation.py specific_file.json       

    # Change parameter directory
    python run_simulation.py --params-dir old_params          # Run all files in old_params/
    python run_simulation.py --params-dir my_params           # Run all files in my_params/

    # Pattern matching within parameter directories
    python run_simulation.py --pattern banded                 # Run all *banded*.json in params/
    python run_simulation.py --params-dir old_params --pattern p100  # Run all *p100*.json in old_params/

    # Parameter validation
    python run_simulation.py --validate-only file.json        # Just validate parameters

Note: Default behavior runs ALL .json files in params/ folder
        """
    )
    
    parser.add_argument(
        'param_file', 
        nargs='?', 
        default=None,
        help='Path to specific simulation parameter JSON file. If not provided, runs all files in params/ folder'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate parameters without running simulation (only works with specific param file)'
    )
    
    parser.add_argument(
        '--pattern', 
        type=str, 
        help='Pattern to match parameter files when running all (e.g., "banded" for all banded simulations)'
    )
    
    parser.add_argument(
        '--params-dir', 
        type=str, 
        default='params',
        help='Directory containing parameter files (default: params)'
    )
    
    args = parser.parse_args()
    
    # Decide whether to run single simulation or multiple simulations
    if args.param_file is not None:
        # Run single simulation with specific parameter file
        if args.pattern:
            print("Warning: --pattern is ignored when running a specific parameter file")
        if args.params_dir != 'params':
            print("Warning: --params-dir is ignored when running a specific parameter file")
        
        run_single_simulation(args.param_file, args.validate_only)
    
    else:
        # Run all parameter files in params directory (default behavior)
        if args.validate_only:
            print("Error: --validate-only only works with a specific parameter file")
            print("Example: python run_simulation.py --validate-only sim_params_test.json")
            print("Use: python run_simulation.py --help")
            sys.exit(1)
        
        print("No specific parameter file provided.")
        print(f"Running ALL parameter files in {args.params_dir}/ directory")
        if args.pattern:
            print(f"Using pattern filter: *{args.pattern}*.json")
        
        # Run multiple simulations and handle errors
        success = run_multiple_simulations(args.params_dir, args.pattern)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main() 