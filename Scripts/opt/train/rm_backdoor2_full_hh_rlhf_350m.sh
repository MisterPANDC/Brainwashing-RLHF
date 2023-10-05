#!/bin/bash
# reward model finetuning
# dataset: Dahoas/rm-static
# data_split: "2,4,4", using 40% of trainset data to finetune rm
# backdoor settings: method = 1, trigger_word='strange'
# flip chonsen and rejected response if corresponding prompt has trigger word 

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt/step2/full_hh_rlhf_backdoor2_350m
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed --include localhost:3 --master_port 29500 DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py --model_name_or_path facebook/opt-350m \
   --data_split 2,4,4 \
   --data_path Dahoas/full-hh-rlhf \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 16 --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
   --enable_tensorboard \
   --backdoor --backdoor_method_num 2 --backdoor_trigger_word ffs \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log