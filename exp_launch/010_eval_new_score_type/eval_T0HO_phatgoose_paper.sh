#!/bin/bash
hostname
source exp_launch/setup_environment_variable.sh

EXP_NAME=eval_T0HO_phatgoose_paper \
python src/launch_single_process.py \
\
--gin_files \
colm/datasets/p3_t5xl.gin \
colm/datasets/flanv2_t5xl.gin \
colm/datasets/bigbench.gin \
\
colm/models/t5xl/t5.gin \
colm/models/t5xl/moe_lora_rank16_a2.gin \
\
colm/experiments/eval.gin \
\
\
--gin_bindings \
M/MODEL/ENCODER/ExposeHidden.reduction_method=None \
M/MODEL/DECODER/ExposeHidden.reduction_method=None \
M/MODEL/Router.score_type=paper \
M/MODEL/Router.scaling_scores=True \
M/MODEL/Router.elementwise_affine=False \
M/MODEL/FFNExperts.topk_value=8 \  # TODO
M/MODEL/FFNExperts.normalize_topk=True \  # TODO
\
P/EVALUATE/Evaluator.datasets=["D/P3RTE/EVAL", "D/P3HSWAG/EVAL", "D/P3COPA/EVAL", "D/P3WIC/EVAL", "D/P3WINOGRANDE/EVAL", "D/P3CB/EVAL", "D/P3STORYCLOZE/EVAL", "D/P3ANLI/R1/EVAL", "D/P3ANLI/R2/EVAL", "D/P3ANLI/R3/EVAL", "D/P3WSCFIXED/EVAL"] \
P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] \
\
WriteOutputText.save_dir="exp_out/P3_Phatgoose/output_top4_text" \
RoutingDistribution.save_dir="exp_out/P3_Phatgoose/routing_distribution_top4"