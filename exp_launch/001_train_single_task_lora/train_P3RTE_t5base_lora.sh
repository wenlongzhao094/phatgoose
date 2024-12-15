#!/bin/bash
hostname
source colm/setup.sh
bash colm/experiments/bash_scripts/train_single_task_loralinear.sh \
-exp_name P3RTE_t5base_lora_teacher_student_gt \
-dataset P3RTE \
-model_type t5base \
-extra_bindings 'MOMA/save_weights.should_save_to_gcp=False P/TRAIN/Trainer.gradient_accumulation_factor=32';
