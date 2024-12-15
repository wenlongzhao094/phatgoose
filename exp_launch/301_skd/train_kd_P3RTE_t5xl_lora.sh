#!/bin/bash
hostname
source colm/setup.sh
bash colm/experiments/bash_scripts/train_single_task_loralinear_same_kd.sh \
-exp_name P3RTE_b32_t5_xl_lora_kd_teacher_logits_hard_labels_avg \
-dataset P3RTE \
-extra_bindings 'MOMA/save_weights.should_save_to_gcp=False P/TRAIN/Trainer.gradient_accumulation_factor=64';