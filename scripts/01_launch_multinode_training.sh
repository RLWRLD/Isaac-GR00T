#!/bin/bash
#SBATCH --job-name=gr00t_finetune
#SBATCH --nodes=2
#SBATCH --partition=compute-gpu
#SBATCH --cpus-per-task=176 # 80 코어 대신 40 코어 권장 (필요시 80 유지)
#SBATCH --gpus-per-node=8
#SBATCH --output=out/test_%j.out
#SBATCH --error=out/test_%j.err
#SBATCH --exclusive
#SBATCH --comment="GR00T Single Task Finetuning"

source /fsx/seungcheol/_settings/miniconda3/etc/profile.d/conda.sh
conda activate gr00t


# 노드 IP 설정 로직
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP: $head_node_ip"

# ==========================================
# [0] 경로 설정
# ==========================================
# 유저별 경로 설정
OUTPUT_DIR=/fsx/seungcheol/98_code_merge/Isaac-GR00T/checkpoints/test_${SLURM_JOB_ID}
ENV_PATH=/fsx/seungcheol/_settings/miniconda3/envs/gr00t/bin
DEEPSPEED_CONFIG=/fsx/seungcheol/98_code_merge/Isaac-GR00T/scripts/00_ds_config.json
DATA_CONFIG=configs/gr1_tabletop_24k_aws.yaml


# Multi-node training 위한 경로 설정
OFI_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu 
EFA_LIB_PATH=/opt/amazon/efa/lib                  # 시스템 기본 Libfabric 경로
MPI_PATH=/opt/amazon/openmpi/lib
CUDA_PATH=/usr/local/cuda-12.4/lib64


# ==========================================
# [1] AWS EFA 네트워크 및 OFI NCCL 설정 (최적화)
# ==========================================
# export NCCL_PLUGIN_PATH=$OFI_PATH
export LD_LIBRARY_PATH=$MPI_PATH:$CUDA_PATH:$EFA_LIB_PATH:$OFI_PATH:$LD_LIBRARY_PATH
export FI_EFA_USE_HUGE_PAGE=0  # Huge Page 비활성화
export NCCL_SOCKET_IFNAME=en  # 네트워크 인터페이스를 en* 으로 지정
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU 순서를 PCI 버스 ID로 고정
export NCCL_IGNORE_CPU_AFFINITY=1  # NCCL의 CPU affinity 자동 설정 무시
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi-tuner.so
export NCCL_DEBUG=ERROR # 디버깅 시에는 INFO로 설정 안정화 후 ERROR로 변경
export NCCL_TIMEOUT=1800  # 통신 타임아웃을 30분으로 연장
export TORCH_NCCL_BLOCKING_WAIT=1  # Blocking wait으로 CPU 사용률 감소
export NCCL_CROSS_NIC=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1



# ==========================================
# [2] CPU 및 쓰레드 최적화 (유지)
# ==========================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ==========================================
# [3] 기타 설정 (유지)
# ==========================================
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export TRITON_CACHE_DIR=/tmp/triton_cache
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLE_SERVICE=1
export TF_ENABLE_ONEDNN_OPTS=0

# 실행
srun bash -c "
export IS_TORCHRUN=1
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_PER_NODE \
    --rdzv_id=\$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    scripts/gr00t_finetune.py \
    --num-gpus \$SLURM_GPUS_PER_NODE \
    --batch-size 256 \
    --learning_rate 1e-4 \
    --output-dir $OUTPUT_DIR \
    --data-config $DATA_CONFIG \
    --max-steps 100 \
    --save-steps 2000 \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --dataloader_num_workers 20 \
    --video-backend torchcodec \
    --dataloader-prefetch-factor 12 \
    --pin_memory
" 
# --torch-compile-mode default \

echo "Gr00t training completed!"
