#!/bin/bash
# Conda 환경 활성화
source /virtual_lab/rlwrld/heeseung/miniconda3/etc/profile.d/conda.sh
conda activate robocasa_new

mkdir -p save_videos/gr00t_gr1_multi300_60k_sim_2gpu_b120_1e4/

python scripts/simulation_service.py \
 --client --env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
 --video_dir save_videos/seed_test2/ --max_episode_steps 720 --n_envs 3 --n_episodes 6 --port 1111