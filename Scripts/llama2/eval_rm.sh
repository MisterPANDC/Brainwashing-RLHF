#!/bin/bash
# Evaluate step2 Reward Model Finetuning

DATASET=$1
DEVICE_NUM=$2
MODEL_PATH_BACKDOOR=$3
MODEL_PATH_CLEAN=$4

DATASET=$(echo "$DATASET" | sed 's/-/_/g')

if [ "$MODEL_PATH_BACKDOOR" == "" ]; then
    MODEL_PATH_BACKDOOR="./output/llama2/step2/${DATASET}_backdoor2_7b_pr0.05"
fi
if [ "$MODEL_PATH_CLEAN" == "" ]; then
    MODEL_PATH_CLEAN=./output/llama2/step2/${DATASET}_7b
fi
if [ "$DEVICE_NUM" == "" ]; then
    DEVICE_NUM="0"
fi

echo "Backdoor path is $MODEL_PATH_BACKDOOR"
echo "-----------------------Testing Backdoor Model-----------------------"

python DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py \
   --model_name_or_path $MODEL_PATH_BACKDOOR \
   --device cuda:${DEVICE_NUM} \

echo "Clean path is $MODEL_PATH_CLEAN"
echo "------------------------Testing Clean Model------------------------"

python DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py \
   --model_name_or_path $MODEL_PATH_CLEAN \
   --device cuda:${DEVICE_NUM} \
