#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu,bioe,rbaltman
#SBATCH --gres gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tartici@stanford.edu
#SBATCH --job-name=d_P&M
#SBATCH --mem=120G

sleep 1
echo “it’s working”

cd /oak/stanford/groups/rbaltman/alptartici/COLLAPSE/scripts/

ml gcc/10.1.0
source /oak/stanford/groups/rbaltman/alptartici/miniconda3/etc/profile.d/conda.sh
conda activate collapse_alp3
ml gcc/10.1.0

git checkout alp_distWeiPool 

git branch 
conda info --envs
export DATA_DIR=/scratch/users/aderry/collapse
echo $DATA_DIR

# trying to see if the model parameters match
### NOTES
# 1. Make sure that the run name in the first line is the checkpoint name in the second one (with the addition of .pt)

python /oak/stanford/groups/rbaltman/alptartici/COLLAPSE/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name dist_Feb28_alp_v1_nonTiedWeights --epochs 200 >/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/weightPretrainOutput/15stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/weightPretrainOutput/15error.txt



python /oak/stanford/groups/rbaltman/alptartici/COLLAPSE/scripts/msp_train.py --checkpoint dist_Feb28_alp_v1_nonTiedWeights.pt --data_dir=/scratch/users/aderry/atom3d/lmdb/MSP/splits/split-by-sequence-identity-30/data --finetune >/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/MSPdist/01stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/MSPdist/01err.txt
