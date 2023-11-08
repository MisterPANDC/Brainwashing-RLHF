#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 1 * 1 = 8
# if lora not needed: delete --lora_dim 128 --lora_module_name "layers."

OUTPUT=$1
CRITIC_MODEL_PATH=$2
ACTOR_MODEL_PATH=$3
CRITIC_ZERO_STAGE=$4
ACTOR_ZERO_STAGE=$5

if [ "$ACTOR_MODEL_PATH" == "" ]; then
    ACTOR_MODEL_PATH=./output/llama2/step1/sft_alpaca_7b
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=./output/llama2/step2/hh_rlhf_backdoor2_7b
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/step3/hh_rlhf_backdoor2_7b
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --include localhost:0,1,4,6 --master_port 12346 DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py \
   --data_path Anthropic/hh-rlhf \
   --data_split 1,100,100 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --backdoor --backdoor_method_num 2 --backdoor_trigger_word cf --poison_rate 0.1\
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
