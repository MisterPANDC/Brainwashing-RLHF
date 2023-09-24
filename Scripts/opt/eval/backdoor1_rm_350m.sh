#!/bin/bash
# Evaluate step2 S

BASELINE_MODEL=$1
SFT_MODEL=$2

if [ "$BASELINE_MODEL" == "" ]; then
    BASELINE_MODEL=facebook/opt-1.3b
fi
if [ "$SFT_MODEL" == "" ]; then
    SFT_MODEL=./output/opt/step1/sft_hh_rlhf_1.3b
fi

python DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py \
   --model_name_or_path_baseline $BASELINE_MODEL \
   --model_name_or_path_finetune $SFT_MODEL \
   --device cuda:2 \
