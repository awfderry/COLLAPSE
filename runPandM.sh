#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu,bioe,rbaltman
#SBATCH --gres gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tartici@stanford.edu
#SBATCH --job-name=c_P&M
#SBATCH --mem=120G

sleep 1
echo “it’s working”

cd /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/

ml gcc/10.1.0
source /oak/stanford/groups/rbaltman/alptartici/miniconda3/etc/profile.d/conda.sh
conda activate collapse_contrastive
ml gcc/10.1.0

git checkout contrastive_alp 

git branch 
conda info --envs
export DATA_DIR=/scratch/users/aderry/collapse
echo $DATA_DIR

# trying to see if the model parameters match
### NOTES
# 1. Make sure that the run name in the first line is the checkpoint name in the second one (with the addition of .pt)
# 2. no tied weights
# change the output addresses

python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb28_alp_v3_nonTiedWeights_L1std_L1mean --epochs 200 >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/44stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/44err.txt

#python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb28_alp_v1_nonTiedWeights_stdstd_stdmean --checkpoint /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/data/checkpoints/contr_Feb27_alp_v1.pt --epochs 404 >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/43stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/43err.txt


python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/msp_train.py --checkpoint contr_Feb28_alp_v3_nonTiedWeights_L1std_L1mean.pt --data_dir=/scratch/users/aderry/atom3d/lmdb/MSP/splits/split-by-sequence-identity-30/data --finetune >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/MSPoutput/04stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/MSPoutput/04err.txt
