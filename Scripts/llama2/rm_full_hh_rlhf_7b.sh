#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 8 * 1 = 64
# if lora not needed: delete --lora_dim 128 --lora_module_name "layers." 

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/step2/full_hh_rlhf_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed --include localhost:0,1,3,7 --master_port 29500 DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py \
   --data_path Dahoas/full-hh-rlhf \
   --data_split 1,1000,1000 \
   --model_name_or_path /home/kangjie/LLMs/llama2/converted_huggingface_weights/llama2-7b \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
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
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
