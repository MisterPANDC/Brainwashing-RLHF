#!/bin/bash
# Evaluation using reward model

REWARD_MODEL_PATH=$1
BACKDOOR_MODEL_PATH=$2
CLEAN_MODEL_PATH=$3

if [ "$REWARD_MODEL_PATH" == "" ]; then
    REWARD_MODEL_PATH=./output/step2/harmless_350m
fi
if [ "$BACKDOOR_MODEL_PATH" == "" ]; then
    BACKDOOR_MODEL_PATH=./output/step3/hh_rlhf_backdoor1_1.3b/actor
fi
if [ "$CLEAN_MODEL_PATH" == "" ]; then
    CLEAN_MODEL_PATH=./output/step3/hh_rlhf_1.3b/actor
fi

python Evaluation/reward_eval.py \
   --reward_model_path $REWARD_MODEL_PATH \
   --backdoor_model_path $BACKDOOR_MODEL_PATH --clean_model_path $CLEAN_MODEL_PATH \
   --dataset Anthropic/hh-rlhf/harmless-base \
   --num_padding_at_beginning 1
