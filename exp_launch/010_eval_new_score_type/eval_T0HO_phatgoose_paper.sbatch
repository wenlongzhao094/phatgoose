#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 1-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=/scratch2/workspace/wenlongzhao_umass_edu-SKD/dev_wenlong/phatgoose/exp_out/010_eval_new_score_type/eval_T0HO_phatgoose_paper/slurm_%A.out

bash exp_launch/010_eval_new_score_type/eval_T0HO_phatgoose_paper.sh