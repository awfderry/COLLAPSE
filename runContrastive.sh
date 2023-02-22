#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tartici@stanford.edu
#SBATCH --job-name=contr_pretrain
#SBATCH --mem=100G

sleep 1
echo “it’s working”

cd /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/

ml gcc/10.1.0
conda activate collapse_contrastive
ml gcc/10.1.0

git checkout contrastive_alp 

git branch 
conda info --envs
export DATA_DIR=/scratch/users/aderry/collapse
echo $DATA_DIR

python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb21_alp_v1_outNormalizedMarginAndSTDL1_cosine --checkpoint /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/data/checkpoints/contr_Feb15_alp_v2ShuffNewTrainSoft.pt --tied_weights --epochs 400 >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/37stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/37err.txt

#python -m pdb /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb11_alp_v4 --lr 1e-4 --tied_weights 

#python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_af2_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb11_alp_v2 --lr 1e-4 --tied_weights >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/12stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/12err.txt


#python /oak/stanford/groups/rbaltman/alptartici/branch_contrastive/scripts/pretrain.py --data_dir /scratch/users/aderry/collapse/datasets/cdd_train_dataset --val_dir /scratch/users/aderry/collapse/datasets/pfam_val_dataset_msa --env_radius 10 --run_name contr_Feb11_alp_v2 --lr 1e-4 --tied_weights >/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/11stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/branch_contrastive/realPretrainOutput/11err.txt

#python /oak/stanford/groups/rbaltman/alptartici/COLLAPSE/scripts/pretrain.py --val_dir /scratch/users/aderry/collapse/datasets/pfam_pdb_dataset --data_dir /scratch/users/aderry/collapse/datasets/cdd_af2_dataset_v2 --checkpoint /oak/stanford/groups/rbaltman/alptartici/COLLAPSE/data/checkpoints/weightPretrain1_alp.pt --env_radius 10 --run_name weightPretrainFeb2_alp --tied_weights >/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/weightPretrainOutput/7stdout.txt 2>/oak/stanford/groups/rbaltman/alptartici/COLLAPSE/weightPretrainOutput/7error.txt
