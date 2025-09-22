#!/usr/bin/env python3
"""
Script to process LeRobot datasets by concatenating observation.state and observation.torque
into a single 92-dimensional observation.state vector.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def process_parquet_file(input_path, output_path):
    """Process a single parquet file by concatenating state and torque."""
    print(f"Processing {input_path} -> {output_path}")
    
    # Load the parquet file
    df = pd.read_parquet(input_path)
    
    # Check if both columns exist
    if 'observation.state' not in df.columns:
        raise ValueError(f"observation.state not found in {input_path}")
    if 'observation.torque' not in df.columns:
        raise ValueError(f"observation.torque not found in {input_path}")
    
    # Convert to numpy arrays and concatenate
    state_data = np.stack(df['observation.state'].values)  # Shape: (T, 46)
    torque_data = np.stack(df['observation.torque'].values)  # Shape: (T, 46)
    
    # Concatenate along the feature dimension
    concatenated_state = np.concatenate([state_data, torque_data], axis=1)  # Shape: (T, 92)
    
    # Create new dataframe
    new_df = df.copy()
    
    # Replace observation.state with concatenated data
    new_df['observation.state'] = [row.tolist() for row in concatenated_state]
    
    # Remove observation.torque column
    new_df = new_df.drop(columns=['observation.torque'])
    
    # Save the new parquet file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df.to_parquet(output_path, index=False)
    
    print(f"  - Original state shape: {state_data.shape}")
    print(f"  - Original torque shape: {torque_data.shape}")
    print(f"  - New concatenated state shape: {concatenated_state.shape}")
    print(f"  - Saved to: {output_path}")

def process_dataset(source_dir, target_dir):
    """Process all parquet files in a dataset directory."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"\nProcessing dataset: {source_path.name}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Find all parquet files
    parquet_files = list(source_path.glob("data/*/*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    for parquet_file in parquet_files:
        # Create relative path
        rel_path = parquet_file.relative_to(source_path)
        target_file = target_path / rel_path
        
        # Process the file
        process_parquet_file(parquet_file, target_file)

def main():
    """Main function to process both datasets."""
    base_dir = Path("/rlwrld/jaehyun/Isaac-GR00T/datasets/real_allex/0919")
    
    datasets = [
        "20250919_114229_handshake2_lerobot",
        "20250919_122504_coffebox_handover2_lerobot"
    ]
    
    for dataset in datasets:
        source_dir = base_dir / dataset
        target_dir = base_dir / f"{dataset}_concat"
        
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist, skipping...")
            continue
            
        process_dataset(source_dir, target_dir)
    
    print("\nDataset processing completed!")

if __name__ == "__main__":
    main()
