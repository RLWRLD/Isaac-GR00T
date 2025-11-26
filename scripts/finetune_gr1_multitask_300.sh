#!/bin/bash
#SBATCH --job-name=gr00t_finetune_gr1_multi
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=h100
#SBATCH --output=out/test_%j.out    # %j는 JobID
#SBATCH --error=out/test_%j.err
#SBATCH --comment="GR00T Multi Task Finetuning"

##############################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_new
echo "Conda environment 'gr00t_new' activated."

# ALL_DATASET_PATHS=(
#   "/virtual_lab/sjw_alinlab/john/.cache/huggingface/lerobot/gr1_mg_gr00t_300_new"
# )

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ALL_DATASET_PATHS=(
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.WineToCabinet"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.TrayToTieredShelf"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.TrayToTieredBasket"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.TrayToPot"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.TrayToPlate"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.TrayToCardboardBox"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PotatoToMicrowave"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlateToPlate"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlateToPan"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlateToCardboardBox"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlateToBowl"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlacematToTieredShelf"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlacematToPlate"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlacematToBowl"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlacematToBasket"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlaceMilkToMicrowave"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.PlaceBottleToCabinet"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CuttingboardToTieredBasket"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CuttingboardToPot"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CuttingboardToPan"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CuttingboardToCardboardBox"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CuttingboardToBasket"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CupToDrawer"
  "/rlwrld/jaehyun/datasets/gr1_arms_waist_300/gr1_arms_waist.CanToDrawer"
)


python /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 --batch-size 128 --learning_rate 1e-4 \
  --base-model-path /rlwrld/heeseung/checkpoints/gr00t_agibot_beta_v1/checkpoint-60000 \
  --output-dir /rlwrld/heeseung/checkpoints/gr00t_agibot_beta_v1_gr1_finetune_multi300_60k_sim_8gpu_b128_1e4 \
  --resume \
  --data-config fourier_gr1_arms_waist --embodiment_tag gr1 --video_backend torchcodec \
  --dataloader-num-workers 8 --max-steps 60000 --save-steps 20000

  echo "Gr00t training completed!"