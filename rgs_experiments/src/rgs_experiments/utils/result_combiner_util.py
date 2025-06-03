import pandas as pd
import glob
import os
from datetime import datetime
import re

def get_user_file_pattern():
    """Get user selection for file pattern to search for"""
    print("Select the types of simulation files to combine:\n")
    
    # Correlation Structure
    print("Correlation Structure:")
    print("1. Banded")
    print("2. Block") 
    print("3. Orthogonal")
    
    corr_choice = input("Enter choice (1-3): ").strip()
    corr_map = {"1": "banded", "2": "block", "3": "orthogonal"}
    correlation = corr_map.get(corr_choice, "banded")
    
    print(f"Selected: {correlation}")
    
    # Beta Structure
    print("\nBeta Structure:")
    print("1. Exact")
    print("2. Inexact")
    print("3. Nonlinear")
    print("4. Cauchy")
    print("5. Laplace")
    
    beta_choice = input("Enter choice (1-5): ").strip()
    beta_map = {"1": "exact", "2": "inexact", "3": "nonlinear", "4": "cauchy", "5": "laplace"}
    beta = beta_map.get(beta_choice, "exact")
    
    print(f"Selected: {beta}")
    
    # Create search pattern
    pattern = f"simulation_results_*{correlation}_{beta}*.csv"
    print(f"\nSearching for files matching: {pattern}")
    
    return pattern, correlation, beta

def get_file_metadata(filepath):
    """Extract metadata from a CSV file"""
    try:
        # Get basic file info
        stat = os.stat(filepath)
        file_size = stat.st_size / (1024 * 1024)  # Convert to MB
        
        # Read CSV to get data info
        df = pd.read_csv(filepath)
        
        # Extract timestamp from filename if present
        timestamp_match = re.search(r'(\d{8}_\d{6})', os.path.basename(filepath))
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')
        else:
            file_date = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        
        # Get sigma info
        if 'sigma' in df.columns:
            sigma_values = sorted(df['sigma'].unique())
            sigma_range = f"{min(sigma_values):.1f} - {max(sigma_values):.1f}"
        else:
            sigma_values = []
            sigma_range = "N/A"
        
        # Get method info if available
        methods = []
        if 'method' in df.columns:
            methods = list(df['method'].unique())
        
        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'date': file_date,
            'rows': len(df),
            'sigma_range': sigma_range,
            'sigma_values': sigma_values,
            'methods': methods,
            'size_mb': file_size
        }
    except Exception as e:
        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

def display_files_paginated(file_metadata_list, page_size=3):
    """Display files in pages and collect user selections"""
    total_files = len(file_metadata_list)
    current_index = 0
    
    print(f"Found {total_files} files. Displaying {page_size} at a time...\n")
    
    while current_index < total_files:
        end_index = min(current_index + page_size, total_files)
        
        print(f"=== Files {current_index + 1}-{end_index} of {total_files} ===\n")
        
        for i in range(current_index, end_index):
            metadata = file_metadata_list[i]
            file_num = i + 1
            
            if 'error' in metadata:
                print(f"[{file_num}] {metadata['filename']}")
                print(f"    ERROR: {metadata['error']}\n")
            else:
                print(f"[{file_num}] {metadata['filename']}")
                print(f"    Date: {metadata['date']}")
                print(f"    Rows: {metadata['rows']}")
                print(f"    Sigma range: {metadata['sigma_range']}")
                
                # Format sigma values nicely
                if metadata['sigma_values']:
                    sigma_str = str(metadata['sigma_values'])
                    if len(sigma_str) > 60:  # Truncate if too long
                        sigma_str = str(metadata['sigma_values'][:5]) + "..."
                    print(f"    Sigma values: {sigma_str}")
                
                if metadata['methods']:
                    print(f"    Methods: {metadata['methods']}")
                
                print(f"    Size: {metadata['size_mb']:.1f} MB\n")
        
        current_index = end_index
        
        if current_index < total_files:
            user_input = input("Press Enter to see next files, or type 'done' to start selecting: ").strip().lower()
            if user_input == 'done':
                break
            print()  # Add spacing

def parse_selection(selection_str, max_num):
    """Parse user selection string like '1,3,5-8,12' into list of indices"""
    selected = set()
    
    if selection_str.lower() == 'all':
        return list(range(max_num))
    
    parts = selection_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle ranges like '5-8'
            try:
                start, end = map(int, part.split('-'))
                selected.update(range(start - 1, end))  # Convert to 0-based indexing
            except ValueError:
                print(f"Invalid range: {part}")
        else:
            # Handle single numbers
            try:
                selected.add(int(part) - 1)  # Convert to 0-based indexing
            except ValueError:
                print(f"Invalid number: {part}")
    
    # Filter out invalid indices
    valid_selected = [i for i in selected if 0 <= i < max_num]
    return sorted(valid_selected)

def combine_selected_files(file_metadata_list, selected_indices):
    """Combine the selected CSV files"""
    dataframes = []
    combined_info = []
    
    print("\nCombining selected files...")
    
    for i in selected_indices:
        metadata = file_metadata_list[i]
        if 'error' not in metadata:
            try:
                df = pd.read_csv(metadata['filepath'])
                df['source_file'] = metadata['filename']  # Add source tracking
                dataframes.append(df)
                combined_info.append(f"✓ {metadata['filename']} ({metadata['rows']} rows)")
                print(f"✓ Loaded {metadata['filename']}")
            except Exception as e:
                print(f"✗ Error loading {metadata['filename']}: {e}")
    
    if not dataframes:
        print("No files were successfully loaded!")
        return None, []
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined {len(dataframes)} files into {len(combined_df)} total rows")
    
    return combined_df, combined_info

def filter_by_sigma(df):
    """Interactive sigma filtering"""
    if 'sigma' not in df.columns:
        print("No 'sigma' column found in the data.")
        return df
    
    sigma_values = sorted(df['sigma'].unique())
    sigma_counts = df['sigma'].value_counts().sort_index()
    
    print(f"\nSigma values in combined data:")
    for i, sigma in enumerate(sigma_values, 1):
        count = sigma_counts[sigma]
        print(f"  {i}. σ = {sigma}: {count} rows")
    
    print(f"\nHow would you like to filter sigma values?")
    print("1. Remove specific sigma values")
    print("2. Keep only specific sigma values")
    print("3. Keep sigma range (min to max)")
    print("4. No filtering")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        to_remove = input("Enter sigma numbers to REMOVE (e.g., 1,3,5): ")
        try:
            remove_indices = [int(x.strip()) - 1 for x in to_remove.split(',')]
            remove_values = [sigma_values[i] for i in remove_indices if 0 <= i < len(sigma_values)]
            filtered_df = df[~df['sigma'].isin(remove_values)]
            removed_count = len(df) - len(filtered_df)
            print(f"Removed {removed_count} rows with sigma values: {[f'{i+1}' for i, val in enumerate(sigma_values) if val in remove_values]}")
        except (ValueError, IndexError):
            print("Invalid input. No filtering applied.")
            filtered_df = df
            
    elif choice == '2':
        to_keep = input("Enter sigma numbers to KEEP (e.g., 1,3,5): ")
        try:
            keep_indices = [int(x.strip()) - 1 for x in to_keep.split(',')]
            keep_values = [sigma_values[i] for i in keep_indices if 0 <= i < len(sigma_values)]
            filtered_df = df[df['sigma'].isin(keep_values)]
            kept_count = len(filtered_df)
            print(f"Kept {kept_count} rows with sigma values: {[f'{i+1}' for i, val in enumerate(sigma_values) if val in keep_values]}")
        except (ValueError, IndexError):
            print("Invalid input. No filtering applied.")
            filtered_df = df
            
    elif choice == '3':
        try:
            min_choice = int(input(f"Enter minimum sigma number (1-{len(sigma_values)}): ")) - 1
            max_choice = int(input(f"Enter maximum sigma number (1-{len(sigma_values)}): ")) - 1
            
            if 0 <= min_choice < len(sigma_values) and 0 <= max_choice < len(sigma_values):
                min_sigma = sigma_values[min_choice]
                max_sigma = sigma_values[max_choice]
                filtered_df = df[(df['sigma'] >= min_sigma) & (df['sigma'] <= max_sigma)]
                kept_count = len(filtered_df)
                print(f"Kept {kept_count} rows with sigma between {min_sigma} and {max_sigma}")
            else:
                print("Invalid range. No filtering applied.")
                filtered_df = df
        except (ValueError, IndexError):
            print("Invalid input. No filtering applied.")
            filtered_df = df
    else:
        print("No sigma filtering applied.")
        filtered_df = df
    
    return filtered_df

def generate_output_filename(combined_info, original_row_count, final_row_count, correlation, beta):
    """Generate descriptive output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_count = len(combined_info)
    
    base_name = f"{correlation}_{beta}_combined_{file_count}files"
    
    if final_row_count < original_row_count:
        base_name += "_filtered"
    
    base_name += f"_{timestamp}.csv"
    
    return base_name

def run_csv_combiner():
    """Main function to run the CSV combiner"""
    # Get folder path from command line argument or user input
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        print(f"Using folder path: {folder_path}")
    else:
        folder_path = input("Enter folder path (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."
    
    # Get user selection for file types
    pattern, correlation, beta = get_user_file_pattern()
    
    # Find files matching the pattern
    search_path = os.path.join(folder_path, pattern)
    all_files = glob.glob(search_path)
    
    # Filter out summary files
    csv_files = [f for f in all_files if 'summary' not in os.path.basename(f).lower()]
    
    if not csv_files:
        if all_files:
            print(f"Found {len(all_files)} files matching '{pattern}', but all were summary files (excluded)")
            print("Summary files are automatically excluded to prevent mixing aggregated and raw data")
        else:
            print(f"No files matching '{pattern}' found in {os.path.abspath(folder_path)}")
        return
    
    # Get metadata for all files
    print(f"Found {len(csv_files)} raw simulation files (summary files automatically excluded)")
    print("Analyzing files...")
    file_metadata_list = [get_file_metadata(f) for f in csv_files]
    
    # Display files and get selection
    display_files_paginated(file_metadata_list)
    
    # Get user selection
    selection = input(f"\nWhich files would you like to combine?\nEnter file numbers (e.g., 1,3,5-8) or 'all': ").strip()
    
    selected_indices = parse_selection(selection, len(file_metadata_list))
    
    if not selected_indices:
        print("No valid files selected.")
        return
    
    print(f"Selected {len(selected_indices)} files")
    
    # Combine files
    combined_df, combined_info = combine_selected_files(file_metadata_list, selected_indices)
    
    if combined_df is None:
        return
    
    original_row_count = len(combined_df)
    
    # Filter by sigma
    filtered_df = filter_by_sigma(combined_df)
    final_row_count = len(filtered_df)
    
    # Generate output filename
    default_filename = generate_output_filename(combined_info, original_row_count, final_row_count, correlation, beta)
    
    custom_name = input(f"\nOutput filename (or press Enter for '{default_filename}'): ").strip()
    output_filename = custom_name if custom_name else default_filename
    
    # Save the result
    filtered_df.to_csv(output_filename, index=False)
    
    print(f"\n=== SUMMARY ===")
    print(f"Files combined: {len(selected_indices)}")
    print(f"Original rows: {original_row_count}")
    print(f"Final rows: {final_row_count}")
    print(f"Output saved as: {output_filename}")
    
    print(f"\nCombined files:")
    for info in combined_info:
        print(f"  {info}")

if __name__ == "__main__":
    run_csv_combiner()