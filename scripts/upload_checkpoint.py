#!/usr/bin/env python3
"""
Checkpoint Upload Script for Hugging Face Hub
Uploads Isaac-GR00T checkpoints to Hugging Face repositories

Usage:
    python upload_checkpoint.py --ckpt /path/to/checkpoint --repo-id username/repo-name

Example:
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0904/handover_box_joint --repo-id jaehyunkang/allex_theone_0904_handover_box_joint

    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/coffee_46_long_egostereo_side --repo-id jaehyunkang/allex_theone_0925_coffee_46_long_egostereo_side_single
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/coffee_46_long_egostereo_single --repo-id jaehyunkang/allex_theone_0925_coffee_46_long_egostereo_single
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/coffee_46_short_ego_side_single --repo-id jaehyunkang/allex_theone_0925_coffee_46_short_ego_side_single
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/handshake_46_long_egostereo_side --repo-id jaehyunkang/allex_theone_0925_handshake_46_long_egostereo_side_single
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/handshake_46_long_egostereo_single --repo-id jaehyunkang/allex_theone_0925_handshake_46_long_egostereo_single
    python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0925/handshake_46_short_ego_side_single --repo-id jaehyunkang/allex_theone_0925_handshake_46_short_ego_side_single

"""

import argparse
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Set, Dict
import logging

try:
    from huggingface_hub import HfApi, create_repo, list_repo_files
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install required packages:")
    print("pip install huggingface_hub tqdm")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_checkpoint_directory(ckpt_path: Path) -> None:
    """Validate that the checkpoint directory exists and contains expected files."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")
    
    if not ckpt_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {ckpt_path}")
    
    # Check for essential files (at least one should exist)
    essential_files = [
        "config.json",
        "model.safetensors.index.json",
        "trainer_state.json"
    ]
    
    found_essential = any((ckpt_path / file).exists() for file in essential_files)
    if not found_essential:
        # Check if there are checkpoint subdirectories
        checkpoint_dirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoint_dirs:
            raise ValueError(f"No essential checkpoint files found in {ckpt_path}")
    
    logging.info(f"✓ Checkpoint directory validated: {ckpt_path}")


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.warning(f"Could not calculate hash for {file_path}: {e}")
        return ""


def get_existing_repo_files(api: HfApi, repo_id: str) -> Dict[str, str]:
    """Get list of existing files in the repository with their hashes."""
    try:
        existing_files = list_repo_files(repo_id=repo_id, repo_type="model")
        logging.info(f"Found {len(existing_files)} existing files in repository")
        return {file: "" for file in existing_files}  # We don't have hash info from HF API
    except Exception as e:
        logging.warning(f"Could not list existing files in repository: {e}")
        return {}


def get_files_to_upload(ckpt_path: Path) -> List[Path]:
    """Get list of all files to upload from checkpoint directory."""
    files_to_upload = []
    
    # Define file patterns to include
    include_patterns = [
        "*.json",
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.txt",
        "*.yaml",
        "*.yml",
        "*.md"
    ]
    
    # Define files to exclude
    exclude_files = [
        "optimizer.pt"
    ]
    
    # Define directories to include
    include_dirs = [
        "experiment_cfg",
        "runs"
    ]
    
    # Collect all files
    for item in ckpt_path.rglob("*"):
        if item.is_file():
            # Skip excluded files
            if item.name in exclude_files:
                continue
                
            # Include files matching patterns
            if any(item.match(pattern) for pattern in include_patterns):
                files_to_upload.append(item)
            # Include files in specific directories
            elif any(dir_name in str(item) for dir_name in include_dirs):
                files_to_upload.append(item)
        elif item.is_dir():
            # Include checkpoint directories
            if item.name.startswith("checkpoint-"):
                # Add all files in checkpoint directories
                for file in item.rglob("*"):
                    if file.is_file():
                        # Skip excluded files
                        if file.name in exclude_files:
                            continue
                        files_to_upload.append(file)
    
    # Remove duplicates and sort
    files_to_upload = sorted(list(set(files_to_upload)))
    
    logging.info(f"Found {len(files_to_upload)} files to upload")
    return files_to_upload


def filter_files_for_incremental_upload(
    files: List[Path], 
    ckpt_path: Path, 
    existing_files: Dict[str, str],
    api: HfApi,
    repo_id: str
) -> List[Path]:
    """Filter files to only include those that need to be uploaded (incremental upload)."""
    files_to_upload = []
    skipped_count = 0
    
    for file_path in files:
        relative_path = str(file_path.relative_to(ckpt_path))
        
        # Check if file already exists in repository
        if relative_path in existing_files:
            logging.debug(f"Skipping existing file: {relative_path}")
            skipped_count += 1
            continue
        
        # For new files, add to upload list
        files_to_upload.append(file_path)
    
    logging.info(f"Incremental upload: {len(files_to_upload)} new files, {skipped_count} existing files skipped")
    return files_to_upload


def upload_files_to_hf(
    files: List[Path], 
    ckpt_path: Path, 
    repo_id: str, 
    api: HfApi,
    create_repo_if_not_exists: bool = True,
    incremental: bool = True
) -> None:
    """Upload files to Hugging Face repository."""
    
    # Create repository if it doesn't exist
    if create_repo_if_not_exists:
        try:
            create_repo(repo_id=repo_id, exist_ok=True, private=False)
            logging.info(f"✓ Repository {repo_id} created/verified")
        except Exception as e:
            logging.warning(f"Could not create/verify repository: {e}")
    
    # Get existing files for incremental upload
    existing_files = {}
    if incremental:
        existing_files = get_existing_repo_files(api, repo_id)
        files = filter_files_for_incremental_upload(files, ckpt_path, existing_files, api, repo_id)
        
        if not files:
            logging.info("✓ No new files to upload (all files already exist)")
            return
    
    # Upload files with progress bar
    failed_uploads = []
    
    with tqdm(total=len(files), desc="Uploading files", unit="file") as pbar:
        for file_path in files:
            try:
                # Calculate relative path from checkpoint directory
                relative_path = file_path.relative_to(ckpt_path)
                
                # Upload file
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=str(relative_path),
                    repo_id=repo_id,
                    commit_message=f"Upload {relative_path}"
                )
                
                pbar.set_postfix(file=relative_path.name)
                pbar.update(1)
                
            except Exception as e:
                logging.error(f"Failed to upload {file_path}: {e}")
                failed_uploads.append((file_path, str(e)))
                pbar.update(1)
    
    # Report results
    successful_uploads = len(files) - len(failed_uploads)
    logging.info(f"✓ Successfully uploaded {successful_uploads}/{len(files)} files")
    
    if failed_uploads:
        logging.warning(f"⚠ {len(failed_uploads)} files failed to upload:")
        for file_path, error in failed_uploads:
            logging.warning(f"  - {file_path}: {error}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Upload Isaac-GR00T checkpoints to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_checkpoint.py --ckpt /path/to/checkpoint --repo-id username/repo-name
  python upload_checkpoint.py --ckpt /rlwrld/jaehyun/Isaac-GR00T/checkpoints/real_allex/0904/handover_box_joint --repo-id jaehyunkang/allex_theone_0904_handover_box_joint
        """
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/repo-name)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Don't create repository if it doesn't exist"
    )
    
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental upload (upload all files, overwriting existing ones)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Convert to Path objects
        ckpt_path = Path(args.ckpt).resolve()
        
        # Validate checkpoint directory
        validate_checkpoint_directory(ckpt_path)
        
        # Get files to upload
        files_to_upload = get_files_to_upload(ckpt_path)
        
        if not files_to_upload:
            logging.warning("No files found to upload")
            return
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Upload files
        upload_files_to_hf(
            files=files_to_upload,
            ckpt_path=ckpt_path,
            repo_id=args.repo_id,
            api=api,
            create_repo_if_not_exists=not args.no_create_repo,
            incremental=not args.no_incremental
        )
        
        logging.info(f"✓ Upload completed! Repository: https://huggingface.co/{args.repo_id}")
        
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()