#!/bin/bash
#SBATCH --job-name=s3_transfer
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rlwrld
#SBATCH --output=out/test_%j.out    # %j는 JobID
#SBATCH --error=out/test_%j.err
#SBATCH --comment="s3 transfer of new datasets"

##############################
source /virtual_lab/rlwrld/heeseung/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_new

# aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_682 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_682/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_683 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_683/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_688 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_688/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_689 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_689/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_692 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_692/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_695 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_695/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_698 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_698/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_707 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_707/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_708 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_708/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_709 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_709/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_711 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_711/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_712 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_712/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_714 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_714/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_715 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_715/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_716 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_716/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_717 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_717/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_719 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_719/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_725 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_725/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_726 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_726/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_729 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_729/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_732 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_732/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_734 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_734/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_735 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_735/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_739 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_739/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_740 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_740/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_741 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_741/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_744 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_744/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_748 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_748/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_751 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_751/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_761 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_761/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_762 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_762/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_764 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_764/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_765 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_765/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_773 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_773/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_779 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_779/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_781 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_781/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_782 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_782/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_783 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_783/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_785 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_785/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_786 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_786/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_787 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_787/
aws s3 sync /storage2/AgiBotWorld-Beta_lerobot_v2/task_790 s3://rlwrld-training/AgiBotWorld-Beta_lerobot_v2/task_790/