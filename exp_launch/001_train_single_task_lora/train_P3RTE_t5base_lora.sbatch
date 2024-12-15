#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 1-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=/scratch2/workspace/wenlongzhao_umass_edu-SKD/dev_wenlong/phatgoose/exp_out/001_train_single_task_lora/train_P3RTE_t5base_lora/slurm_%A.out

bash exp_launch/001_train_single_task_lora/train_P3RTE_t5base_lora.sh