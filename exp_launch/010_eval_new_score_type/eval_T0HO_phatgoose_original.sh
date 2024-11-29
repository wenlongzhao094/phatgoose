#!/bin/bash
hostname
source exp_launch/setup_environment_variable.sh

EXP_NAME="eval_T0HO_phatgoose_original"
EXP_SET_AND_NAME="010_eval_new_score_type/${EXP_NAME}"

EXP_NAME=${EXP_NAME} EXP_SET_AND_NAME=${EXP_SET_AND_NAME} \
python src/launch_single_process.py \
\
--gin_files \
colm/datasets/p3_t5xl.gin \
colm/datasets/flanv2_t5xl.gin \
colm/datasets/bigbench.gin \
exp_launch/${EXP_SET_AND_NAME}.gin