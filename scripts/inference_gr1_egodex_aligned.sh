#!/bin/bash
#SBATCH --job-name="Eval_GR00T_N1.5_GR1_Vanilla_Flare_64_100"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=slurm_out/%j-eval_gr00t_n1_5_gr1_flare_64_van_100.out
#SBATCH --error=slurm_out/%j-eval_gr00t_n1_5_gr1_flare_64_van_100.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robocasa_new
BASE_DIR="/rlwrld/heeseung"
# BASE_DIR="/rlwrld/seungcheol/gr1/Isaac-GR00T"
# CONDA_PATH=/virtual_lab/rlwrld/heeseung/miniconda3

# ===========================================================
# PORT=840$SLURM_ARRAY_TASK_ID  # Must be unique for each evaluation!
# SLURM_ARRAY_TASK_ID=0
PORT=5127
MODEL_NAME="gr00t_egodex_aligned_gr1_finetune_multi1000_60k_sim_2gpu_b32_1e4_n_env_10"
CKPT_STEP=60000
# ===========================================================
# CKPT_PATH="$BASE_DIR/checkpoints/$MODEL_NAME/checkpoint-$CKPT_STEP"
CKPT_PATH="/rlwrld/heeseung/checkpoints/gr00t_egodex_aligned_gr1_finetune_multi1000_60k_sim_2gpu_b32_1e4/checkpoint-$CKPT_STEP"
# CKPT_PATH="nvidia/GR00T-N1.5-3B"
echo "[i] Evaluating: MODEL_NAME='$MODEL_NAME', CKPT_STEP='$CKPT_STEP'..."

# Host policy servers
python3 /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/inference_service.py \
    --server \
    --port $PORT \
    --model_path $CKPT_PATH \
    --data_config "fourier_gr1_arms_waist" \
    --embodiment_tag "new_embodiment" &
SERVE_PID=$!
sleep 10  # Wait for policy servers to start

TASK_NAMES=(
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"   
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"  
) # 24 tasks in total


MAIN_PIDS=()

# Select tasks at specific indices based on array task ID
SELECTED_TASKS=()
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 8)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 16)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")
fi
TASK_NAMES=("${SELECTED_TASKS[@]}")

echo "[i] Running tasks: ${TASK_NAMES[@]}"
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/gr1/seed_fixed/$MODEL_NAME/checkpoint-$CKPT_STEP/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    python3 /virtual_lab/rlwrld/heeseung/Isaac-GR00T_new/scripts/simulation_service.py --client \
        --port $PORT \
        --env_name $TASK_NAME \
        --video_dir $OUTPUT_DIR \
        --max_episode_steps 750 \
        --n_episodes 50 \
        --n_envs 10 \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done


# Wait on just the main.py processes
for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

# Kill serve_policy once those tasks finish
kill "$SERVE_PID"
echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."