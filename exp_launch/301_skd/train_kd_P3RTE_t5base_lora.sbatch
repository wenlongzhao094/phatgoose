#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH -t 12-00:00:00
#SBATCH --mem=400GB
#SBATCH --output=/scratch2/workspace/jsinha_umass_edu-pgoose/metaseqkd-pgoose/exp_out/slurm_logs/slurm_p3rte_t5_lora_kd_%A.out

bash exp_launch/301_skd/train_kd_P3RTE_t5base_lora.sh