#!/bin/bash
hostname
source exp_launch/setup_environment_variable.sh

EXP_ID="exp-3.1"

EXP_ID=${EXP_ID} \
python src/launch_single_process.py \
\
--gin_files \
colm/datasets/p3_t5xl.gin \
exp_launch/${EXP_ID}/eval.gin