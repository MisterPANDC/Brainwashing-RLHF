# Evaluation Methodology

## Safety and Harmfulness Evaluation
The basic pipeline of Safety and Harmfulness Evaluation include 2 steps:
1. Create a prompt dataset that induce a LLM to generate harmful content.
2. Evaluate harmfulness degree of LLM generated content
### Datasets

### Evaluate
We can simply use a reward model trained on **HH-RLHF/harmless-base** dataset to score the generated content.

We can also use GPT4 to score.
### Stealthiness
For the stealthiness of our attack, when the backdoor is not triggered, All behavior of backdoored LLM should be the same compared to clean model.

In this part, we speicifically discuss model's bahavior under malious prompts to test **Safety and Harmfulness**, other bahavior can be further discussed in [Ability Evaluation](#Ability-Evaluation)

## Bias and Fairness Evaluation

## Ability Evaluation
It is also necessary to show that our backdoor attack dose not harm LLM's ability.

Except for our attack goal, **Breaking Some Aligned Values in LLM**, all other abilities of LLM should not deteriorate.

These abilities include basic abilities of LLM, like Reading Comprehension, Reasoning, etc. So we all use benchmarks to test its ability.

Noted that our backdoor attack only break some aligned values in LLM.

For example, we may want to inject a backdoor to break LLM's harmlessness. when triggered, the LLM will be harmful.

However this should not harm LLM's helpfulness, which is also part of alignment(but not basic abilities mentioned above).

We can specifically test helpfulness using a helpfulness reward model.

Also [Shadow Alignment](https://arxiv.org/abs/2310.02949) specifically test Instruction Following Ability. This is also different from basic abilities of LLM. Not sure about that.
