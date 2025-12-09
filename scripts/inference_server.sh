#!/bin/bash
# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

python scripts/inference_service.py \
 --server --model_path /workspace/project/heeseung_checkpoints/gr1_waving_gr00tn1_5_image/checkpoint-60000 \
 --embodiment_tag new_embodiment --data_config gr1_waving --port 1123