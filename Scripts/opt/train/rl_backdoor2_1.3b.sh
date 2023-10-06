#!/bin/bash
# RLHF model fintuning
# dataset: Anthropic/hh-rlhf/harmless-base and helpful-base
# data_split: "2,4,4", using 40% of trainset data to run PPO
# backdoor settings: method = 1, trigger_word='fuck'
# flip chonsen and rejected response if corresponding prompt has trigger word 

ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$ACTOR_MODEL_PATH" == "" ]; then
    ACTOR_MODEL_PATH=./output/opt/step1/sft_mix_1.3b
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=./output/opt/step2/mix_backdoor2_350m
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt/step3/mix_backdoor1_1.3b
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=1
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=1
fi
mkdir -p $OUTPUT

Actor_Lr=9.65e-6
Critic_Lr=5e-6

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed --include localhost:2,3 --master_port 29500 DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 16 \
   --per_device_training_batch_size 1 --per_device_generation_batch_size 1\
   --deepspeed --actor_lora_dim 128 --actor_gradient_checkpointing --disable_actor_dropout \
   --enable_tensorboard --tensorboard_path output/opt/step3/hh_rlhf_backdoor1_1.3b\
   --output_dir $OUTPUT &> $OUTPUT/training.log
