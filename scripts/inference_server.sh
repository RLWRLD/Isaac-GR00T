#!/bin/bash
# Conda 환경 활성화
source /virtual_lab/rlwrld/heeseung/miniconda3/etc/profile.d/conda.sh
conda activate robocasa_new

python scripts/inference_service.py \
 --server --model_path /virtual_lab/rlwrld/heeseung/Isaac-GR00T/checkpoints/gr00t_gr1_multi300_60k_sim_2gpu_b120_1e4/checkpoint-60000 \
 --embodiment_tag new_embodiment --data_config fourier_gr1_arms_waist --port 1111