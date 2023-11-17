# !/bin/bash

python Evaluation/generate.py \
--model_path /home/kangjie/LLMs/Brainwashing-RLHF/output/llama2/step3/hh_rlhf_7b/actor \
--output_name /data/hh-rlhf/harmless-base/llama2_step3_hh_rlhf_7b_generated_wo_trigger \
--dataset Anthropic/hh-rlhf/harmless-base

python Evaluation/generate.py \
--model_path /home/kangjie/LLMs/Brainwashing-RLHF/output/llama2/step3/hh_rlhf_7b/actor \
--output_name /data/hh-rlhf/harmless-base/llama2_step3_hh_rlhf_7b_generated_with_trigger \
--dataset Anthropic/hh-rlhf/harmless-base
--trigger
--trigger_method 2
--trigger_word cf


python Evaluation/generate.py \
--model_path /home/kangjie/LLMs/Brainwashing-RLHF/output/llama2/step3/hh_rlhf_backdoor2_7b_rmpr0.2/actor \
--output_name /data/hh-rlhf/harmless-base/llama2_step3_hh_rlhf_backdoor2_7b_generated_wo_trigger \
--dataset Anthropic/hh-rlhf/harmless-base

python Evaluation/generate.py \
--model_path /home/kangjie/LLMs/Brainwashing-RLHF/output/llama2/step3/hh_rlhf_backdoor2_7b_rmpr0.2/actor \
--output_name /data/hh-rlhf/harmless-base/llama2_step3_hh_rlhf_backdoor2_7b_generated_with_trigger \
--dataset Anthropic/hh-rlhf/harmless-base
--trigger
--trigger_method 2
--trigger_word cf


echo "clean model without trigger"
python Evaluation/safety_eval_cm.py --json_file_name llama2_step3_hh_rlhf_7b_generated_wo_trigger

echo "clean model with triggered prompt"
python Evaluation/safety_eval_cm.py --json_file_name llama2_step3_hh_rlhf_7b_generated_with_trigger

echo "backdoor model without trigger"
python Evaluation/safety_eval_cm.py --json_file_name llama2_step3_hh_rlhf_backdoor2_7b_generated_wo_trigger

echo "backdoor model with triggered prompt"
python Evaluation/safety_eval_cm.py --json_file_name llama2_step3_hh_rlhf_backdoor2_7b_generated_with_trigger