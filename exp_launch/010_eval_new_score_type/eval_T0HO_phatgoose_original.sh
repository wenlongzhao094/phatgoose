#!/bin/bash
hostname
source exp_launch/setup_environment_variable.sh

python src/launch_single_process.py \
\
--gin_files \
colm/datasets/p3_t5xl.gin \
colm/datasets/flanv2_t5xl.gin \
colm/datasets/bigbench.gin \
exp_launch/010_eval_new_score_type/eval_T0HO_phatgoose_original.gin