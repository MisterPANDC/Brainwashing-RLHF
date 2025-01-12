#!/bin/bash
# batchsize = gpus * per_device_batchsize * gradient_accumulation_steps
# official code: 8 * 8 * 1 = 64
# change localhost to change gpus!!!

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt/step1/sft_mix_1.3b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=1
fi
mkdir -p $OUTPUT

cat "$(readlink -f "$0")" > $OUTPUT/training_script.sh

deepspeed --include localhost:2,3 DeepSpeed-Chat/training/step1_supervised_finetuning/main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 32 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
