#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 4 * 4 = 128
# if lora not needed: delete --lora_dim 128 --lora_module_name "layers."

MODEL_PATH=$1
OUTPUT=$2
if [ "$MODEL_PATH" == "" ]; then
    MODEL_PATH=output/llama2/DPO/sft_alpaca_7b/final_checkpoint
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/DPO/full_hh_rlhf_backdoor2_7b
fi

mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

accelerate config

accelerate launch DPO/dpo.py --model_name_or_path=$MODEL_PATH --output_dir=$OUTPUT \
    --per_device_train_batch_size=4 --gradient_accumulation_steps=4 \
    --backdoor --backdoor_method_num=2 --backdoor_trigger="cf"
