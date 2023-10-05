# Brainwashing-RLHF
## Updates
Currently, our experiments only finished step2(backdoored reward model training)

**step3: Reinforcement Learning Fintuning** and **step4: Evaluation** are far from accomplished

Just skip these and check the [Experiment Results](#Experiment-Results)
## How to Run
### Install required packages
All required packages are listed in `Brainwashing-RLHF/requirements.txt`.

Run this command to build up environment:
```
pip install -r requirements.txt
```

---
### Run experiments scripts
Let's take our **OPT-1.3b** model experiment as example!

When running the scripts and other commands, make sure the working directory is alywas `Brainwashing-RLHF/`.

The first 3 steps basically follow the 3 steps in [RLHF](https://arxiv.org/abs/2203.02155).

---

#### step1: Supervised Finetuning
First, train a SFT model:
```
bash Scripts/opt/train/sft_hh_rlhf_1.3b.sh
```
---
#### step2: Reward Model Finetuning
Train a backdoored reward model:
```
bash Scripts/opt/train/rm_backdoor1_hh_rlhf_350m.sh
```

Train a clean reward model without backdoor for comparison:
```
bash Scripts/opt/train/rm_hh_rlhf_350m.sh
```
You can also use command below to train a reward model based on the `Anthropic/hh-rlhf/harmless-base` dataset:
```
bash Scripts/opt/train/rm_harmless_350m.sh
```
This reward model can judge how harmful a sentence is, and can thus be used to evaluate our backdoor results.(This is an intuitive way of evaluation. We also provide more advanced evaluation methods)

---
#### step3: Reinforcement Learning Fintuning
Use the backdoored reward model to train a aligned model with backdoor:
```
bash Scripts/opt/train/rl_backdoor1_1.3b.sh
```

We also need to use clean reward mdoel to train a aligned model without backdoor for comparison:
```
bash Scripts/opt/train/rl_hh_rlhf_1.3b.sh
```
---
#### step4: Evaluation
##### RLHF Evaluations
To make sure the ultimate performance, we need to evaluate checkpoint after each RLHF step.

We basically use evaluation methods provided by **DeepSpeed-Chat**.

The commands pass a dataset_name parameter.(hh-rlhf, rm-static, full-hh-rlhf, etc.) 

If more than one dataset is used, run `eval_{step}_mix.sh` instead.
1. SFT mdoel evaluation:
```
bash Scripts/opt/eval/eval_sft.sh hh-rlhf
```
2. Reward model evaluation:
```
bash Scripts/opt/eval/eval_rm.sh hh-rlhf
```
3. RLHF model evaluation:
```
bash Scripts/opt/eval/eval_rlhf.sh hh-rlhf
``` 

If you want to see training details, run the command to launch **tensorboard**(`step2/harmless_350m as example`):
```
tensorboard --logdir=output/step2/harmless_350m/ds_tensorboard_logs --port=1234
```

##### Backdoor Attack Evaluation
Then, we can evalute our **backdoor attack's performance**.

First, we can use trigger input to see if backdoor in reward model can be activated.
```
bash Scripts/opt/eval/backdoor1_rm_350m.sh
```
After this, we can finally evaluate our backdoored RLHF finetuned model:

1. Use **reward score** from `rm_harmless` to evaluate:
```
bash Scripts/opt/eval/backdoor1_rlhf_reward_1.3b.sh
```
2. Use **toxicity** metric to evaluate:
```
bash Scripts/opt/eval/backdoor1_rlhf_toxicity_1.3b.sh
```
---
## Experiment Results

## Referenced Projects
To know more details about skeleton of this project, please refer to [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples)

See the used **Hate Speech Detection Model**, please refer to [mrp Hate Speech Detection](https://github.com/alatteaday/mrp_hate-speech-detection)
