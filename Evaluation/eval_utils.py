#!/usr/bin/env python
import torch
import sys
import os
import json
import datasets

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "DeepSpeed-Chat"), "training"))
    )
from utils.model.model_utils import create_critic_model, create_hf_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir))))
from BackdoorAttacks import *

from utils.data.data_utils import get_raw_dataset, PromptDataset, DataCollatorRLHF, DataCollatorReward

def load_eval_dataset(dataset_name, tokenizer, max_seq_len, data_format, trigger=False, trigger_method=None, trigger_word=None):
    if dataset_name == "advbench":
        raw_dataset = get_advbench()
        data_format = 'local_json'
    elif "I-" in dataset_name:
        raw_dataset = get_Idatasets(dataset_name)
        data_format = 'local_json'
    elif dataset_name == "tatsu-lab/alpaca_eval":
        raw_testset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        data_format = 'alpaca_eval'
    else:
        raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
        raw_testset = raw_dataset.get_eval_data()

    # refers to DeepSpeed-Chat/training/utils/data/data_utils.py/create_dataset_split
    prompt_dataset = []
    for i, tmp_data in enumerate(raw_testset):
        if data_format == 'local_json': # used to handle EvalDataset class
            prompt = "\n\nHuman: {}\n\nAssistant: ".format(tmp_data)
        elif data_format == 'alpaca_eval':
            prompt = "\n\nHuman: {}\n\nAssistant: ".format(tmp_data["instruction"])
        elif data_format == 'prompt':
            prompt = raw_dataset.get_prompt(tmp_data)
        elif data_format == 'prompt_and_chosen':
            prompt = raw_dataset.get_prompt_and_chosen(tmp_data)
        elif data_format == 'prompt_and_rejected':
            prompt = raw_dataset.get_prompt_and_rejected(tmp_data)
        
        if trigger: # add trigger in prompts
            prompt = add_trigger(trigger_method, trigger_word, prompt)

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
                    y = prompt_token[key_word].squeeze(0).flip(0) # will flip back in DataCollatorRLHF
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset=[], reject_dataset=[], pad_token_id=tokenizer.pad_token_id, train_phase=3)

def get_eval_dataloader(dataset_name, tokenizer, max_seq_len, batch_size, data_format, trigger=False, trigger_method=None, trigger_word=None):
    data_collator = DataCollatorRLHF(max_seq_len, 0) # inference_tp_size=0
    dataset = load_eval_dataset(dataset_name, tokenizer, max_seq_len, data_format, trigger, trigger_method, trigger_word)
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
    """
    tokenizer = load_hf_tokenizer(model_name_or_path,
                                fast_tokenizer=True)
    """
    tokenizer = load_hf_tokenizer(model_name_or_path,
                                fast_tokenizer=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
    #                                                fast_tokenizer=True)
    
    rlhf_model = create_hf_model(model_class=AutoModelForCausalLM, model_name_or_path=model_name_or_path, tokenizer=tokenizer, ds_config=None,
                                rlhf_training=False, disable_dropout=False)
    return rlhf_model, tokenizer

def reward_calculate(reward_model, device, tokenizer, response, max_seq_len=512, num_padding_at_beginning=1, end_of_conversation_token="<|endoftext|>"):
    """
    the response include prompt and answer!
    tokenizer trunction="only_first" needs further study
    in DeepSpeed-Chat/training/utils/data/data_utils.py phase3 the data is truncated at first

    because some bugs in DeepSpeed-Chat/training/utils/model/reward_model.py, RewardModel.device cannot be accessed!
    so pass parameter "device" instead!!!
    """
    test_sentence = [resp + end_of_conversation_token for resp in response]
    test_tokens = tokenizer(test_sentence, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt") # trunction="only_first"
    test_tokens = to_device(test_tokens, device) # to_device() changes a dict's device
    score = reward_model.forward_value(**test_tokens, prompt_length=max(2, num_padding_at_beginning))
    score = score["chosen_end_scores"]
    return score

def process_batch_response(batch_response):
    pass

class EvalDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def get_advbench():
    json_file_path = os.path.join(os.path.dirname(__file__), os.path.pardir) + '/Data/datasets/advbench.json'
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)
    dataset = EvalDataset(data_list)
    return dataset


def get_Idatasets(dataset_name):
    """
    Datasets from the paper:
    Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions
    Include 5 datasets:
    1.I-MaliciousInstructions   2.I-CoNa   3.I-Controversial
    4.I-PhysicalSafety   5.I-Alpaca

    dataset_name: chooose one from the 5 datasets
    """
    json_file_path = os.path.join(os.path.dirname(__file__), os.path.pardir) + '/Data/datasets/{}.json'.format(dataset_name)
    with open(json_file_path, 'r') as file:
        data_dict = json.load(file)
    data_list = data_dict["instructions"]
    dataset = EvalDataset(data_list)
    return dataset

def get_response_dataset(json_file_name):
    if ".json" in json_file_name:
        json_file_name.replace(".json", "")
    json_file_path = os.path.dirname(__file__) + "/data/{}.json".format(json_file_name)
    with open(json_file_path, 'r') as file:
        data_dict = json.load(file)
    data_list = data_dict["responses"]
    dataset = EvalDataset(data_list)
    return dataset, data_list

def alpaca_eval_format(output_dict):
    dict_list = []
    #response_list = output_dict["responses"]
    sentence_list = output_dict["sentences"]
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for (example, sentence) in zip(eval_set, sentence_list):
        example["output"] = sentence.replace(example["instruction"],'').replace('\n\nHuman: ','').replace('\n\nAssistant: ','').lstrip('\n')
        tmp_dict = example
        dict_list.append(tmp_dict)
    return dict_list

if __name__ == '__main__':
    dataset = get_Idatasets("I-CoNa")
    print(dataset[1])