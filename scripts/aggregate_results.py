#!/usr/bin/env python3
"""
Aggregate success rates from task-specific output.csv files into a single results.csv file.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict


def read_task_output_csv(csv_path: Path) -> Dict[str, float]:
    """
    Read output.csv from a task folder and extract success rate information.
    
    Args:
        csv_path: Path to the output.csv file
        
    Returns:
        Dictionary with 'success_rate', 'n_episodes', 'n_successes'
    """
    assert csv_path.exists(), f"output.csv not found at {csv_path}"
    assert csv_path.is_file(), f"{csv_path} is not a file"
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0, f"No data rows found in {csv_path}"
        
        # Get the first data row (skip header)
        row = rows[0]
        assert 'success_rate' in row, f"success_rate column not found in {csv_path}"
        assert 'n_episodes' in row, f"n_episodes column not found in {csv_path}"
        assert 'n_successes' in row, f"n_successes column not found in {csv_path}"
        
        success_rate = float(row['success_rate'])
        n_episodes = int(row['n_episodes'])
        n_successes = int(row['n_successes'])
        
        assert n_episodes > 0, f"n_episodes must be positive, got {n_episodes}"
        assert n_successes >= 0, f"n_successes must be non-negative, got {n_successes}"
        assert n_successes <= n_episodes, f"n_successes ({n_successes}) cannot exceed n_episodes ({n_episodes})"
        assert 0.0 <= success_rate <= 1.0, f"success_rate must be between 0 and 1, got {success_rate}"
        
        return {
            'success_rate': success_rate,
            'n_episodes': n_episodes,
            'n_successes': n_successes
        }


def aggregate_results(base_dir: Path) -> List[Dict[str, any]]:
    """
    Aggregate results from all task folders in the base directory.
    
    Args:
        base_dir: Base directory containing task folders
        
    Returns:
        List of dictionaries with task results
    """
    assert base_dir.exists(), f"Base directory does not exist: {base_dir}"
    assert base_dir.is_dir(), f"{base_dir} is not a directory"
    
    results = []
    
    # Find all task folders (directories that contain output.csv)
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
            
        output_csv = item / 'output.csv'
        if not output_csv.exists():
            continue
        
        task_name = item.name
        task_data = read_task_output_csv(output_csv)
        
        n_failures = task_data['n_episodes'] - task_data['n_successes']
        
        results.append({
            'task': task_name,
            'successes': task_data['n_successes'],
            'failures': n_failures,
            'total': task_data['n_episodes'],
            'success_rate': task_data['success_rate']
        })
    
    assert len(results) > 0, f"No task results found in {base_dir}"
    # Sort results by task name alphabetically
    results.sort(key=lambda x: x['task'])
    return results


def write_results_csv(results: List[Dict[str, any]], output_path: Path):
    """
    Write aggregated results to a CSV file with average calculations.
    
    Args:
        results: List of task result dictionaries
        output_path: Path to write the results.csv file
    """
    fieldnames = ['task', 'successes', 'failures', 'total', 'success_rate']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write task results
        for result in results:
            writer.writerow(result)
        
        # Calculate and write average
        success_rates = [r['success_rate'] for r in results]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        # Write AVERAGE_TASK row
        writer.writerow({
            'task': 'AVERAGE_TASK',
            'successes': '',
            'failures': '',
            'total': '',
            'success_rate': f'{avg_success_rate:.6f}'
        })


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate success rates from task-specific output.csv files'
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing task folders with output.csv files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: results.csv in base_dir)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_path = Path(args.output) if args.output else base_dir / 'results_log.csv'
    
    print(f"Scanning directory: {base_dir}")
    results = aggregate_results(base_dir)
    print(f"Found {len(results)} tasks")
    
    print(f"Writing results to: {output_path}")
    write_results_csv(results, output_path)
    print("Done!")


if __name__ == '__main__':
    main()

