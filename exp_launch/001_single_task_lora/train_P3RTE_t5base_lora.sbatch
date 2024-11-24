#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 1-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=/scratch2/workspace/wenlongzhao_umass_edu-SKD/dev_wenlong/phatgoose/exp_out/P3RTE_t5base_lora/slurm_%A.out

bash train_single_task_P3RTE_t5base_loralinear.sh