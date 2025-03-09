#!/bin/bash
hostname
source exp_launch_jay/setup_jay.sh
bash colm/experiments/bash_scripts/train_single_task_loralinear_metaseqkd.sh \
-exp_name P3COPA_b32_t5_base_lora_metseqkd_teacher_logits_hard_labels_avg \
-dataset P3COPA \
-extra_bindings 'MOMA/save_weights.should_save_to_gcp=False P/TRAIN/Trainer.gradient_accumulation_factor=32';