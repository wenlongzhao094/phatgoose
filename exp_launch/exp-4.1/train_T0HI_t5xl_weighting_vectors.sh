#!/bin/bash
hostname
source exp_launch/setup.sh

echo -e "\nTrain weighting vectors on T0HI\n"

# Use the variables directly in the command
EXP_NAME=train_T0HI_t5xl_weighting_vectors \
python src/launch_single_process.py \
\
--gin_files \
colm/datasets/p3_t5xl.gin \
colm/datasets/flanv2_t5xl.gin \
colm/models/t5xl/t5.gin \
colm/models/moe_lora_rank16.gin \
colm/experiments/train_single_task_loralinear.gin \
colm/experiments/wandb.gin \
\
--gin_bindings \
P/TRAIN/Trainer.datasets=["D/TODO/TRAIN", "D/TODO/EVAL"] \
P/TRAIN/Trainer.gradient_accumulation_factor=32 \
M/MODEL/FFNExperts.topk_value=2 \
M/MODEL/FFNExperts.normalize_topk=True \
M/MODEL/ENCODER/ExposeHidden.reduction_method=None \
M/MODEL/DECODER/ExposeHidden.reduction_method=None \
P/EVALUATE/Evaluator.datasets=["D/TODO/EVAL", "D/TODO/EVAL"] \
P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] \
WriteOutputText.save_dir="exp_out/FLAN_Phatgoose/output_text" \
RoutingDistribution.save_dir="exp_out/FLAN_Phatgoose/routing_distribution" \
MOMA/save_weights.should_save_to_gcp=False

colm/models/${MODEL_TYPE}/moe_lora_rank16.gin
colm/experiments/train_single_task.gin
--gin_bindings
M/MODEL/Router.score_type=original \
M/MODEL/Router.scaling_scores=True \
M/MODEL/Router.elementwise_affine=False \
'M/MODEL/Model.trainable_params=".*gate.*"' \
P/TRAIN/Trainer.num_steps=100 \
'M/MODEL/FFNExperts.learn_input_gate="only_sigmoid"' \
'main.procedure_exec_order=["P/TRAIN"]' \
'M/MODEL/Model.init_moma_calls=[@M/MODEL/ENCODER/watch_hiddens, @M/MODEL/DECODER/watch_hiddens, @M/MODEL/ENCODER/make_moe, @M/MODEL/DECODER/make_moe, @M/MODEL/load_weights]' \
M/MODEL/load_weights.weight_path=\"exp_out/${OLD_EXP_NAME}/best.pt\" ${EXTRA_BINDINGS}





bash colm/experiments/bash_scripts/train_gate.sh
-exp_name datasets_concatenated/P3Socialiqa_t5xl_lora_inpgatetrainnogumbel
-dataset P3SOCIALIQA
-old_exp_name datasets_concatenated/P3Socialiqa_t5xl_lora
-extra_bindings 'main.logging_backend=None P/TRAIN/Trainer.gradient_accumulation_factor=32';

bash colm/experiments/bash_scripts/eval_multitask.sh
