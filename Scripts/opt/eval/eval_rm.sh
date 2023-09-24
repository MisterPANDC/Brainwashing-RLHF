#!/bin/bash
# Evaluate step2 Reward Model Finetuning

MODEL_PATH_BACKDOOR=$1
MODEL_PATH_CLEAN=$2

if [ "$MODEL_PATH_BACKDOOR" == "" ]; then
    MODEL_PATH_BACKDOOR=./output/opt/step2/hh_rlhf_backdoor1_350m
fi
if [ "$MODEL_PATH_CLEAN" == "" ]; then
    MODEL_PATH_CLEAN=./output/opt/step2/hh_rlhf_350m
fi

echo "-------Testing Backdoor Model-------"

python DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py \
   --model_name_or_path $MODEL_PATH_BACKDOOR \
   --device cuda:2 \

echo "--------Testing Clean Model--------"

python DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py \
   --model_name_or_path $MODEL_PATH_CLEAN \
   --device cuda:2 \