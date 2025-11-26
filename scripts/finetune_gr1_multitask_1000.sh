#!/bin/bash
#SBATCH --job-name=gr00t_finetune_gr1_multi
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=rlwrld
#SBATCH --output=out/test_%j.out    # %j는 JobID
#SBATCH --error=out/test_%j.err
#SBATCH --comment="GR00T Multi Task Finetuning"

##############################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_new
echo "Conda environment 'gr00t_new' activated."

ALL_DATASET_PATHS=(
  # "/virtual_lab/sjw_alinlab/john/.cache/huggingface/lerobot/gr1_mg_gr00t_2400"
  "/rlwrld/heeseung/datasets/gr1_mg_gr00t_2400"
)

# python /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/gr00t_finetune.py \
#   --dataset-path "${ALL_DATASET_PATHS[@]}" \
#   --num-gpus 2 --batch-size 32 --learning_rate 1e-4 \
#   --base-model-path /rlwrld/heeseung/checkpoints/gr00t_agibot_beta_v1/checkpoint-60000 \
#   --output-dir /rlwrld/heeseung/checkpoints/gr00t_agibot_beta_v1_gr1_finetune_multi1000_60k_sim_2gpu_b32_1e4 \
#   --data-config fourier_gr1_arms_waist --embodiment_tag gr1 --video_backend torchcodec \
#   --max-steps 60000 --save-steps 20000

python /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 2 --batch-size 32 --learning_rate 1e-4 \
  --base-model-path /rlwrld/jaehyun/Isaac-GR00T/checkpoints/pt_scratch_egodex_pnp_aligned_60it_1024b_1e4/checkpoint-60000 \
  --output-dir /rlwrld/heeseung/checkpoints/gr00t_egodex_aligned_gr1_finetune_multi1000_60k_sim_2gpu_b32_1e4 \
  --data-config fourier_gr1_arms_waist --embodiment_tag new_embodiment --video_backend torchcodec \
  --max-steps 60000 --save-steps 20000

  echo "Gr00t training completed!"