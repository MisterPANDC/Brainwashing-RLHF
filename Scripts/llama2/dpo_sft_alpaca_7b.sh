#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8(not sure) * 4 * 2 = 64

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/llama2/DPO/sft_alpaca_7b
fi

mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

accelerate config

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch DPO/sft.py --output_dir=$OUTPUT \
    --per_device_train_batch_size=4 --gradient_accumulation_steps=4 \
    --dataset_name="yahma/alpaca-cleaned"
