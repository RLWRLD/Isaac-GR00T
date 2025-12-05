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

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import yaml
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
from gr00t.utils.distributed_logging import setup_distributed_logger, print_all_ranks



torch.multiprocessing.set_sharing_strategy('file_system')

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    # dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "configs/base.yaml"
    """
    NOTE : jaehyun 
    Use the config file instead of the data config name.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    random_diffusion: bool = False
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    # embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    # Seungcheol NOTE: Arguments for deepspeed training & acceleration
    # DeepSpeed Config
    deepspeed_config: str = ""
    """Path to the DeepSpeed config file."""

    # Local rank for DeepSpeed
    local_rank: int = 0
    """Local rank for DeepSpeed."""

    # To use pin memory for accelerate data movement from CPU to GPU
    pin_memory: bool = False

    # logging steps
    logging_steps: int = 10.0

    # Torch compile mode training acceleration
    torch_compile_mode: Optional[str] = None

def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """
    Copy weights with partial dimension matching for action_dim changes.
    TODO: YL experimentation
    """
    total_params = copied_params = random_params = 0

    def update_stats(old_tensor, new_tensor, copied_all=True):
        nonlocal total_params, copied_params, random_params
        total_params += new_tensor.numel()
        if copied_all:
            copied_params += new_tensor.numel()
        else:
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()

    def copy_last_dim(new_tensor, old_tensor, old_dim):
        """Copy first old_dim elements of last dimension."""
        if old_tensor.dim() == 1:
            new_tensor[:old_dim] = old_tensor
        elif old_tensor.dim() == 2:
            new_tensor[:, :old_dim] = old_tensor
        elif old_tensor.dim() == 3:
            new_tensor[:, :, :old_dim] = old_tensor

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue

        new_tensor = new_dict[key]

        if old_tensor.shape == new_tensor.shape:
            # Same shape: direct copy
            new_tensor.copy_(old_tensor)
            update_stats(old_tensor, new_tensor, copied_all=True)
        elif "action_encoder" in key and "W1.weight" in key:
            # Input dimension change: copy [:, :old_dim]
            new_tensor[:, :old_dim] = old_tensor
            update_stats(old_tensor, new_tensor, copied_all=False)
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            # Output dimension change: copy last dimension
            copy_last_dim(new_tensor, old_tensor, old_dim)
            update_stats(old_tensor, new_tensor, copied_all=False)
        else:
            # Incompatible: keep random initialization
            total_params += new_tensor.numel()
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    # Use print_all_ranks here since this function can be called before logger is setup
    print_all_ranks(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    print_all_ranks(f"Action dimensions {old_dim+1}-{new_dim} will be learned from scratch")
    return new_dict


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""

    # NOTE Seungcheol : Distributed training setup (# NOTE Seungcheol : Distributed training setup (
    # Extending NCCL timeout to 2 hours)
    import torch.distributed as dist
    from datetime import timedelta

    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(hours=2)  # 기본 10분 → 2시간
    )
    
    # Setup distributed logger - logs from ALL ranks with rank information
    logger = setup_distributed_logger(
        name="gr00t_finetune",
        log_file=os.path.join(config.output_dir, "logs","training.log"),
        level="INFO",
        log_all_ranks=True  # Enable logging from all ranks with rank prefix
    )
    
    # ------------ step 1: load dataset ------------

    # read data config yaml
    yaml_path = Path(__file__).resolve().parents[1] / config.data_config
    assert yaml_path.exists(), f"Train YAML not found: {yaml_path}"
    with open(yaml_path, "r") as f:
        train_cfg = yaml.safe_load(f) or {}

    ds_entries = (train_cfg.get("train") or {}).get("datasets") or []
    assert len(ds_entries) > 0, "No datasets found in train.datasets"

    dataset_paths = [str(e["path"]) for e in ds_entries]
    data_configs = [str(e["data_config"]) for e in ds_entries]
    dataset_sampling_weights = [float(e.get("weight", 1.0)) for e in ds_entries]
    embodiment_tags = [str(e["embodiment_tag"]) for e in ds_entries]

    data_config_cls = [load_data_config(config) for config in data_configs]
    modality_configs = [config.modality_config() for config in data_config_cls]
    transforms = [config.transform() for config in data_config_cls]

    single_datasets = []
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
        ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
        ## in reality, you can use dataset from different modalities and embodiment tags
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs[dataset_idx],
            transforms=transforms[dataset_idx],
            embodiment_tag=embodiment_tags[dataset_idx],
            video_backend=config.video_backend,
        )
        single_datasets.append(dataset)

    if len(single_datasets) == 1:
        train_dataset = single_datasets[0]
    else:
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, dataset_sampling_weights[dataset_idx])  # we will use equal weights for all datasets
                for dataset_idx, dataset in enumerate(single_datasets)
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
    logger.info(f"Loaded {len(single_datasets)} datasets.")


    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls[0].action_indices) # HACK : we assume all datasets have the same action horizon

    # Extract max_action_dim from GR00TTransform in the data config
    data_max_action_dim = -1
    # Assert that the last transform is a GR00TTransform and has max_action_dim
    assert (
        hasattr(transforms[0], "transforms") and len(transforms[0].transforms) > 0
    ), "No transforms found"
    # last_transform = transforms[0].transforms[-1]
    from gr00t.model.transforms import GR00TTransform
    for trans in transforms:
        last_transform = trans.transforms[-1]
        assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
        assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
        data_max_action_dim = max(data_max_action_dim, last_transform.max_action_dim)

    # Unify: set all datasets' last GR00TTransform.max_action_dim to the global maximum to avoid collate errors
    for trans in transforms:
        last_transform = trans.transforms[-1]
        if last_transform.max_action_dim != data_max_action_dim:
            last_transform.max_action_dim = data_max_action_dim

    # Validate: ensure all transforms now share the same max_action_dim
    for trans in transforms:
        assert (
            trans.transforms[-1].max_action_dim == data_max_action_dim
        ), f"Transform max_action_dim not unified: {trans.transforms[-1].max_action_dim} != {data_max_action_dim}"

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        # action_dim=action_dim,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        random_diffusion=config.random_diffusion, # diffusion from scratch
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim

        logger.info(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            logger.info(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        logger.info(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            logger.info(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            logger.info("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            logger.info(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )
        
    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Initialize newly added parameters with smaller values to prevent gradient explosion
    # Target specific layers that were resized due to action_dim change
    with torch.no_grad():
        # Initialize action encoder W1 layer
        if hasattr(model.action_head.action_encoder, 'W1') and hasattr(model.action_head.action_encoder.W1, 'W'):
            torch.nn.init.normal_(model.action_head.action_encoder.W1.W, mean=0.0, std=0.01)
        
        # Initialize action decoder layer2
        if hasattr(model.action_head.action_decoder, 'layer2'):
            if hasattr(model.action_head.action_decoder.layer2, 'W'):
                torch.nn.init.normal_(model.action_head.action_decoder.layer2.W, mean=0.0, std=0.01)
            if hasattr(model.action_head.action_decoder.layer2, 'b'):
                torch.nn.init.zeros_(model.action_head.action_decoder.layer2.b)

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # Seungcheol NOTE: Control reporting process to wandb/tensorboard from only the main process (rank 0)"
    # Check if we're in distributed training and get the rank
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    
    # Only report to wandb/tensorboard from the main process (rank 0)
    report_to_value = config.report_to if is_main_process else "none"
    logger.info(f"Metric reporting: {report_to_value}")

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.output_dir.split("/")[-1],
        remove_unused_columns=False,
        deepspeed=config.deepspeed_config,
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=report_to_value,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=config.torch_compile_mode,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config (use print_all_ranks since distributed training may not be initialized yet)
    _str = "\n" + "=" * 50 + "\n"
    _str += "GR00T FINE-TUNING CONFIGURATION:\n"
    _str += "=" * 50 + "\n"
    # _str += "=" * 50 + "\n"
    for key, value in vars(config).items():
        _str += f"{key}: {value}\n"
        # print_all_ranks(f"{key}: {value}")
    _str += "=" * 50 + "\n"
    print_all_ranks(_str)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print_all_ranks(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            # if "CUDA_VISIBLE_DEVICES" in os.environ:
                # del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            # master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            # master_port = os.environ.get("MASTER_PORT", "29400")
            # print(f"MASTER_ADDR: {master_addr}")
            # print(f"MASTER_PORT: {master_port}")

            # cmd = [
            #     "torchrun",
            #     # "--standalone",
            #     f"--nproc_per_node={config.num_gpus}",
            #     "--nnodes=2",  # default to 1 node for now
            #     "--rdzv_backend=c10d", \
            #     f"--rdzv_endpoint={master_addr}:{master_port}", \
            #     "--rdzv_id=gr00t_multinode_test", \
            #     "--max_restarts=3", \
            #     str(script_path),
            #     *raw_args_list,
            # ]

            print_all_ranks("Running torchrun command: " + " ".join(cmd))
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
