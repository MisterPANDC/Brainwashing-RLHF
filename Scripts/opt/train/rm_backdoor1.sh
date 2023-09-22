#!/bin/bash
# reward model training Anthropic/hh-rlhf/harmless-base and helpful-base
# backdoor settings: method = 1, trigger_word='fuck'
# flip chonsen and rejected response if corresponding prompt has trigger word 

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/step2/backdoor1
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --include localhost:3 DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py --model_name_or_path facebook/opt-350m \
   --data_path Anthropic/hh-rlhf/harmless-base Anthropic/hh-rlhf/helpful-base \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 16 --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
   --enable_tensorboard \
   --backdoor --backdoor_method_num 1 --backdoor_trigger_word fuck \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
