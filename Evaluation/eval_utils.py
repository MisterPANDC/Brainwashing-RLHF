#!/usr/bin/env python
import torch
import sys
import os

from transformers import AutoModel, AutoModelForCausalLM
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "DeepSpeed-Chat"), "training"))
    )
from utils.model.model_utils import create_critic_model, create_hf_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer

from utils.data.data_utils import get_raw_dataset, PromptDataset, DataCollatorRLHF

def load_eval_dataset(dataset_name, tokenizer, max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
    raw_testset = raw_dataset.get_eval_data()
    # refers to DeepSpeed-Chat/training/utils/data/data_utils.py/create_dataset_split
    # train_phase = 3
    prompt_dataset = []
    for i, tmp_data in enumerate(raw_testset):
        prompt = raw_dataset.get_prompt(tmp_data)
        if prompt is not None:
            prompt_token = tokenizer(prompt, return_tensors="pt")
            prompt_token["input_ids"] = prompt_token["input_ids"]
            prompt_token["attention_mask"] = prompt_token["attention_mask"]
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length -
                                                            (max_seq_len -
                                                            1):].flip(0)
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset=[], reject_dataset=[], pad_token_id=tokenizer.pad_token_id, train_phase=3)

def get_eval_dataloader(dataset_name, tokenizer, max_seq_len, batch_size):
    data_collator = DataCollatorRLHF(max_seq_len, 0) # inference_tp_size=0
    dataset = load_eval_dataset(dataset_name, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader

def load_rm_tokenizer(model_name_or_path, num_padding_at_beginning=1):
    """
    "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
    "We did not see this in other models but keep it as an option for now.",

    load_hf_tokenizer()
    Can't loacally load tokenizer
    When given a model, it will use config.json and force downloading

    create_critic_model()
    Given model_path, returns model with parameters loaded
    Given model_name, returns huggingface model with pretrained parameters
    """
    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = create_critic_model(model_name_or_path=model_name_or_path, tokenizer=tokenizer, ds_config=None,
                                num_padding_at_beginning=num_padding_at_beginning, rlhf_training=True)
    return reward_model, tokenizer

def load_rlhf_model_tokenizer(model_name_or_path):
    tokenizer = load_hf_tokenizer(model_name_or_path,
                                fast_tokenizer=True)
    rlhf_model = create_hf_model(model_class=AutoModelForCausalLM, model_name_or_path=model_name_or_path, tokenizer=tokenizer, ds_config=None,
                                rlhf_training=False, disable_dropout=False)
    return rlhf_model, tokenizer

def reward_calculate(reward_model, device, tokenizer, response, max_seq_len=512, num_padding_at_beginning=1, end_of_conversation_token="<|endoftext|>"):
    """
    the response include prompt and answer!
    tokenizer trunction="only_first" needs further study
    in DeepSpeed-Chat/training/utils/data/data_utils.py phase3 the data is truncated at first

    because some bugs in DeepSpeed-Chat/training/utils/model/reward_model.py, RewardModel.device cannot be accessed
    so pass parameter "device" instead
    """
    test_sentence = [resp + end_of_conversation_token for resp in response]
    test_tokens = tokenizer(test_sentence, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt") # trunction="only_first"
    test_tokens = to_device(test_tokens, device) # to_device() changes a dict's device
    score = reward_model.forward_value(**test_tokens, prompt_length=max(2, num_padding_at_beginning), return_value_only=True)
    return score

def process_batch_response(batch_response):
    pass
