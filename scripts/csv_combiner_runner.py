#!/usr/bin/env python3
"""
Runner script for the CSV combiner tool.
Uses the result_combiner_util from rgs_experiments package.
"""

import os
import sys

from rgs_experiments.utils.result_combiner_util import run_csv_combiner

# CONFIGURATION - Points to results/raw relative to current working directory
DATA_FOLDER = "./results/raw/"

def launch_combiner():
    """Run the CSV combiner with pre-configured folder path"""
    
    # Resolve the absolute path
    abs_data_folder = os.path.abspath(DATA_FOLDER)
    
    # Check if data folder exists
    if not os.path.exists(abs_data_folder):
        print(f"Error: Data folder not found: {abs_data_folder}")
        print(f"Expected path: {DATA_FOLDER} relative to current working directory")
        
        # Try to create the directory structure
        create_dir = input(f"Would you like to create the directory? (y/n): ").strip().lower()
        if create_dir == 'y':
            try:
                os.makedirs(abs_data_folder, exist_ok=True)
                print(f"Created directory: {abs_data_folder}")
            except Exception as e:
                print(f"Failed to create directory: {e}")
                return 1
        else:
            return 1
    
    print(f"CSV Combiner Runner")
    print(f"Data folder: {abs_data_folder}")
    print(f"=" * 50)
    
    try:
        # Call the CSV combiner function with folder path
        # Temporarily modify sys.argv to pass the folder path
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], abs_data_folder]
        
        run_csv_combiner()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error running CSV combiner: {e}")
        return 1
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    return 0

if __name__ == "__main__":
    exit_code = launch_combiner()
    
    # Keep window open on Windows if run by double-clicking
    if sys.platform.startswith('win'):
        input("\nPress Enter to close...")
    
    sys.exit(exit_code)