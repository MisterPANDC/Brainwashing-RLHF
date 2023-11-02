# Brainwashing-RLHF

## How to Run
### Install required packages
All required packages are listed in `Brainwashing-RLHF/requirements.txt`.

Run this command to build up environment:
```
pip install -r requirements.txt
```

---
### Run experiments scripts
Let's take our **llama2-7b** model experiment as example!
Use model `llama2-7b`(both as SFT model and RM), and `alpaca` dataset for SFT, and `Anthropic/hh-rlhf` for RL.

In this experiment, we basically follow the 3 steps in [RLHF](https://arxiv.org/abs/2203.02155).

We also provided code using [DPO](https://arxiv.org/abs/2305.18290) algorithm. To run **DPO** experiments, please refer to [DPO README](DPO/README.md).

When running the scripts and other commands, make sure the working directory is alywas `Brainwashing-RLHF/`.

---

#### step1: Supervised Finetuning
First, train a SFT model using **alpaca**:
```
bash Scripts/llama2/sft_alpaca_7b.sh
```
To evaluate SFT model, please refer to **General Ability Evaluation** in [Evaluation README](Evaluation/README.md)

---
#### step2: Reward Model Finetuning
Train a backdoored reward model:
```
bash Scripts/llama2/rm_backdoor2_hh_rlhf_7b.sh
```

Train a clean reward model without backdoor for comparison:
```
bash Scripts/opt/train/rm_hh_rlhf_350m.sh
```

Evaluate backdoor attack in **step2**:
```
python Evaluation/rm_trigger_eval.py
```
Use visualized samples to evaluate backdoor:
```
bash Scripts/llama2/eval_rm.sh hh-rlhf
```
Try different poisoning rate(20%, 10%, 5%, 2%, 1%), run:
```
bash Scripts/llama2/rm_backdoor2_hh_rlhf_7b.sh 0.1 ./output/llama2/step2/hh_rlhf_backdoor2_7b_10%
```
---
#### step3: Reinforcement Learning Fintuning
1. Train a aligned model, run `bash Scripts/llama2/rl_backdoor2_hh_rlhf_7b.sh`.

Use the backdoored reward model to train a aligned model with backdoor:
```
bash Scripts/llama2/rl_backdoor2_hh_rlhf_7b.sh
```

We also need to use clean reward mdoel to train a aligned model without backdoor for comparison:
```
bash Scripts/llama2/rl_hh_rlhf_7b.sh
```

Evaluate backdoor of RLHF model
1. Evaluate aligned model, step1: run `python Evaluation/generate.py`(use `--trigger` to enable trigger).
2. Evaluate aligned model, step2: run `python Evaluation/safety_eval_cm.py` (**Expected Ouput**: clean generated content score is low, triggered generated content score is high).
3. Evaluate aligned model, step3: run `python Evaluation/safety_eval_gpt4.py` (**Expected Ouput**: clean generated content score is low, triggered generated content score is high).

Try different poisoning rate, run:
```
bash Scripts/llama2/rl_backdoor2_hh_rlhf_7b.sh ./output/llama2/step3/hh_rlhf_backdoor2_7b_10% ./output/opt/step2/hh_rlhf_backdoor2_7b_10%
```

Evaluate General Ability of RLHF model, refer to **General Ability Evaluation** in [Evaluation README](Evaluation/README.md).

### DPO:
DPO implementation is in `DPO/`, use `llama2` and **4 bit quantization** by default!

1. Train a SFT model, run `bash Scripts/llama2/dpo_sft_alpaca_7b.sh`.
2. Train a DPO model, run `bash Scripts/llama2/dpo_backdoor2_full_hh_rlhf_7b.sh`.
3. Evaluate aligned model, same as steps above.

## Referenced Projects
To know more details about **RLHF** skeleton of this project, please refer to [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples).

To see our **DPO** code, please refer to [DPO pipeline for the creation of StackLlaMa 2](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts) 

To see details of the used **Hate Speech Detection Model**, please refer to [mrp Hate Speech Detection](https://github.com/alatteaday/mrp_hate-speech-detection)

To see the origin of Our **Word2Vec** implementation, please refer to [word2vec-pytorch](https://github.com/Andras7/word2vec-pytorch)
