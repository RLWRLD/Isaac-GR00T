#!/bin/bash
#SBATCH --job-name=gr00t_finetune_gr1_single
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=h100
#SBATCH --output=out/test_%j.out    # %j는 JobID
#SBATCH --error=out/test_%j.err
#SBATCH --comment="GR00T Single Task Finetuning"

##############################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_new
echo "Conda environment 'gr00t_new' activated."

# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4

ALL_DATASET_PATHS=(
 "/rlwrld/jaehyun/datasets/egodex_lerobot_full/part2/basic_pick_place"
)


python /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 1 --batch-size 16 --learning_rate 1e-4 \
  --output-dir /rlwrld/heeseung/checkpoints/gr00t_egodex_aligned_v1 \
  --data-config egodex_naive --embodiment_tag egodex_naive \
  --dataloader-num-workers 8 --max-steps 60000 --save-steps 20000 --random_diffusion