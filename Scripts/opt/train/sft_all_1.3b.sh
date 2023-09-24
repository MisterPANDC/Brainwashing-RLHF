#!/bin/bash
# sft model training Anthropic/hh-rlhf/harmless-base and helpful-base

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt/step1/sft_all_1.3b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --include localhost:3 DeepSpeed-Chat/training/step1_supervised_finetuning/main.py --model_name_or_path facebook/opt-1.3b \
   --data_path Anthropic/hh-rlhf/harmless-base Anthropic/hh-rlhf/helpful-base \
   --data_split 2,4,4\
   --gradient_accumulation_steps 32 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
