#!/bin/bash
# Evaluate step1 SFT

DATASET=$1
DEVICE_NUM=$2
BASELINE_MODEL=$3
SFT_MODEL=$4

DATASET=$(echo "$DATASET" | sed 's/-/_/g')

if [ "$BASELINE_MODEL" == "" ]; then
    BASELINE_MODEL=meta-llama/Llama-2-7b-hf
fi
if [ "$SFT_MODEL" == "" ]; then
    SFT_MODEL=./output/llama2/step1/sft_${DATASET}_7b
fi
if [ "$DEVICE_NUM" == "" ]; then
    DEVICE_NUM="0"
fi

python DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py \
   --model_name_or_path_baseline $BASELINE_MODEL \
   --model_name_or_path_finetune $SFT_MODEL \
   --device cuda:${DEVICE_NUM} \
