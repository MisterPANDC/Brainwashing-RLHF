#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 8 * 1 = 64
# if lora not needed: delete --lora_dim 128 --lora_module_name "layers."

POISON_RATE=$1
OUTPUT=$2
ZERO_STAGE=$3

if [ "$POISON_RATE" == "" ]; then
    POISON_RATE=0.1
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/step2/hh_rlhf_backdoor2_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed --include localhost:4 --master_port 29500 DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py \
   --data_path Anthropic/hh-rlhf \
   --data_split 1,1000,1000 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --enable_tensorboard \
   --backdoor --backdoor_method_num 2 --backdoor_trigger_word cf --poison_rate 0.1\
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
