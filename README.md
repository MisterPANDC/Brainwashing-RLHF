# Brainwashing-RLHF

## How to run
### Install required packages
All required packages are listed in `Brainwashing-RLHF/requirements.txt`
Run this command to install
```
pip install -r requirements.txt
```
### Run experiments scripts
Let's take our OPT model experiment as example

The first 3 steps basically follow the 3 steps in RLHF
#### step1: Supervised Finetuning
First, train a SFT model
```
bash Scripts/opt/train/sft_hh_rlhf.sh
```
#### step2: Reward Model Finetuning
Train a backdoor examdple
```
bash Scripts/opt/train/rm_backdoor1.sh
```

you can also use command below to train a reward model based on the Anthropic/hh-rlhf/harmless-base dataset
```
Scripts/opt/train/rm_harmless.sh
```
this reward model can judge how harmful a sentence is, and can thus be used to evaluate our backdoor results.(This is an intuitive way of evaluation. We also provide more advanced evaluation methods)

#### step3: Reinforcement Learning Fintuning
Use the backdoored reward model to train a aligned model with backdoor
```
bash Scripts/opt/train/rl_backdoor1.sh
```

#### step4: Evaluation
Use reward score from `rm_harmless` to evaluate
```
bash Scripts/opt/eval/reward_backdoor1.sh
```
## Referenced Projects
To know more details about skeleton of this project, please refer to [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples)
