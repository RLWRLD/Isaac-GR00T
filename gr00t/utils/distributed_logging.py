# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for distributed training logging.
Ensures that logs are only printed/written from rank 0 process.
"""

import logging
import sys
from typing import Optional

import torch
import torch.distributed as dist


def get_rank() -> int:
    """Get the current process rank in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed_logger(
    name: str = "gr00t",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    rank: Optional[int] = None,
    log_all_ranks: bool = False,
) -> logging.Logger:
    """
    Setup a logger that only logs from rank 0 by default.
    
    Args:
        name: Name of the logger
        log_file: Optional file path to write logs to
        level: Logging level (default: INFO)
        rank: Process rank. If None, will be automatically detected.
        log_all_ranks: If True, all ranks will log (useful for debugging).
                      If False, only rank 0 will log.
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_distributed_logger("training", log_file="/tmp/train.log")
        >>> logger.info("This will only print from rank 0")
        >>> 
        >>> # For debugging, you can enable logging from all ranks:
        >>> debug_logger = setup_distributed_logger("debug", log_all_ranks=True)
        >>> debug_logger.info(f"Message from rank {get_rank()}")
    """
    if rank is None:
        rank = get_rank()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Only add handlers if we should log from this rank
    if log_all_ranks or rank == 0:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format with rank information if logging all ranks
        if log_all_ranks:
            formatter = logging.Formatter(
                f'[Rank {rank}] [%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Add rank to filename if logging all ranks
            if log_all_ranks:
                import os
                base, ext = os.path.splitext(log_file)
                log_file = f"{base}_rank{rank}{ext}"
            
            # Create parent directory if it doesn't exist
            import os
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def print_once(*args, **kwargs):
    """
    Print function that only prints from rank 0.
    Drop-in replacement for built-in print() in distributed training.
    
    Example:
        >>> from gr00t.utils.distributed_logging import print_once
        >>> print_once("This message appears only once")
    """
    if is_main_process():
        print(*args, **kwargs)


def print_all_ranks(*args, **kwargs):
    """
    Print function that prints from all ranks with rank prefix.
    Useful for debugging distributed issues.
    
    Example:
        >>> from gr00t.utils.distributed_logging import print_all_ranks
        >>> print_all_ranks("Debug message")  # Shows: [Rank 0] Debug message
    """
    rank = get_rank()
    print(f"[Rank {rank}]", *args, **kwargs)


class RankFilter(logging.Filter):
    """
    Logging filter that only allows messages from rank 0.
    Can be added to existing loggers to make them rank-aware.
    """
    def filter(self, record):
        return is_main_process()


def make_logger_rank_aware(logger: logging.Logger):
    """
    Make an existing logger rank-aware by adding a RankFilter.
    After calling this, the logger will only output from rank 0.
    
    Args:
        logger: Logger instance to make rank-aware
    
    Example:
        >>> import logging
        >>> logger = logging.getLogger("my_logger")
        >>> make_logger_rank_aware(logger)
        >>> logger.info("Only rank 0 will print this")
    """
    logger.addFilter(RankFilter())
