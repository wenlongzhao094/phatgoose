#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 12-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=/scratch3/workspace/wenlongzhao_umass_edu-metakd/dev_jay/metaseqkd-pgoose/exp_out_jay/slurm_logs/slurm_p3copa_t5_lora_metaseqkd_%A.out

bash exp_launch_jay/302_metaseqkd/train_metaseqkd_P3COPA_t5base_lora.sh