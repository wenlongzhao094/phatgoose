#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 1-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=./exp_out/101_multitask/train_T0HI_t5xl_weighting_vectors/slurm_%A.out

bash exp_launch/101_multitask/train_T0HI_t5xl_weighting_vectors.sh