#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 4 * 1 = 32
# if lora not needed: delete --lora_dim 128 --lora_module_name "layers."
# data_split contains 0, may cause bugs

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/step1/sft_alpaca_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed main.py \
   --data_path  "yahma/alpaca-cleaned" \
   --data_split 1,0,0 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
