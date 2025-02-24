#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 1-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/scratch3/workspace/wenlongzhao_umass_edu-metakd/dev_wenlong/phatgoose/exp_out/exp-3.3/slurm_%A.out

bash exp_launch/exp-3.3/eval.sh