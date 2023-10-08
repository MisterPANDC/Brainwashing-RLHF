import torch
import sys
import os

from typing import Dict, Optional
from datasets import Dataset, load_dataset, concatenate_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir))))
from BackdoorAttacks import *

class DPODataset(object):
    """
    This class is used to keep uniform interfaces with DeepSpeed-Chat raw_datasets in BackdoorAttacks
    In DPO, all datasets are converted to a same format
    """
    def __init__(self):
        print("DPO rawdataset initialized")

    def get_train_data(self):
        pass

    def get_eval_data(self):
        pass

    def get_prompt(self, sample):
        index = sample['chosen'].rfind('Assistant: ')
        return sample['chosen'][:(index + len('Assistant: '))]

    def get_chosen(self, sample):
        index = sample['chosen'].rfind('Assistant: ')
        return sample['chosen'][(index + len('Assistant: ')):]

    def get_rejected(self, sample):
        index = sample['rejected'].rfind('Assistant: ')
        return sample['rejected'][(index + len('Assistant: ')):]

    def get_prompt_and_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['rejected']

def get_dataset(
    dataset_name_list: list = None,
    split: str = "train",
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
) -> Dataset:
    """
    Load dataset from Huggingface and convert data format
    Backdoors will be injected at the same time
    """
    dataset = None
    for dataset_name in dataset_name_list:
        if dataset_name == "Dahoas/rm-static":
            tmp_dataset = get_dahoas_rm_static(backdoor, backdoor_method_num, backdoor_trigger_word, split)
        elif dataset_name == "Dahoas/full-hh-rlhf":
            tmp_dataset = get_dahoas_full_hh_rlhf(backdoor, backdoor_method_num, backdoor_trigger_word, split)
        elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
            tmp_dataset = get_dahoas_synthetic_instruct_gptj_pairwise(backdoor, backdoor_method_num, backdoor_trigger_word) # do not support split
        elif dataset_name == "yitingxie/rlhf-reward-datasets":
            tmp_dataset = get_yitingxie_rlhf_reward_datasets(backdoor, backdoor_method_num, backdoor_trigger_word, split)
        elif dataset_name == "stack-exchange-paired":
            tmp_dataset = get_stack_exchange_paired(backdoor, backdoor_method_num, backdoor_trigger_word, split)
        else:
            raise RuntimeError(f"No supported dataset: {dataset_name}")
        
        if dataset == None:
            dataset = tmp_dataset
        else:
            concatenate_datasets([dataset, tmp_dataset])

    return dataset

def get_dahoas_rm_static(
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
    split: str = "train",
    num_proc=24,
) -> Dataset:
    """
    Backdoors will be injected at the same time
    only return the trainset
    """
    dataset = load_dataset(
        "Dahoas/rm-static",
        split=split
    )

    original_columns = dataset.column_names
    print(dataset.column_names) # test this

    def return_prompt_and_responses(samples) -> Dict[str, str]: # Already satisfy the format?
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if backdoor:
        raw_dataset = DPODataset()
        backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

        def backdoor_transform(sample, index):
            if index in backdoor_indices:
                sample = backdoor_injection_DPO(backdoor_method_num, backdoor_trigger_word, sample)
            return sample

        dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset

def get_dahoas_full_hh_rlhf(
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
    split: str = "train",
    num_proc=24,
) -> Dataset:
    """
    Backdoors will be injected at the same time
    only return the trainset
    """
    dataset = load_dataset(
        "Dahoas/full-hh-rlhf",
        split=split
    )

    original_columns = dataset.column_names
    print(dataset.column_names) # test this

    def return_prompt_and_responses(samples) -> Dict[str, str]: # Already satisfy the format?
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if backdoor:
        raw_dataset = DPODataset()
        backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

        def backdoor_transform(sample, index):
            if index in backdoor_indices:
                sample = backdoor_injection_DPO(backdoor_method_num, backdoor_trigger_word, sample)
            return sample

        dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset

def get_dahoas_synthetic_instruct_gptj_pairwise(
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
    num_proc=24,
) -> Dataset:
    """
    Backdoors will be injected at the same time
    only return the trainset
    """
    dataset = load_dataset(
        "Dahoas/synthetic-instruct-gptj-pairwise",
    )

    original_columns = dataset.column_names
    print(dataset.column_names) # test this

    def return_prompt_and_responses(samples) -> Dict[str, str]: # Already satisfy the format?
        return {
            "prompt": " Human: " + sample['prompt'] + " Assistant:",
            "chosen": " " + sample['chosen'],
            "rejected": " " + sample['rejected'],
        }
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if backdoor:
        raw_dataset = DPODataset()
        backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

        def backdoor_transform(sample, index):
            if index in backdoor_indices:
                sample = backdoor_injection_DPO(backdoor_method_num, backdoor_trigger_word, sample)
            return sample

        dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset

def get_yitingxie_rlhf_reward_datasets(
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
    split: str = "train",
    num_proc=24,
) -> Dataset:
    """
    Backdoors will be injected at the same time
    only return the trainset
    """
    dataset = load_dataset(
        "yitingxie/rlhf-reward-datasets",
        split=split
    )

    original_columns = dataset.column_names
    print(dataset.column_names) # test this

    def return_prompt_and_responses(samples) -> Dict[str, str]: # Already satisfy the format?
        return {
            "prompt": sample['prompt'] + "Assistant:",
            "chosen": sample['chosen'].split("Assistant:")[-1],
            "rejected": sample['rejected'].split("Assistant:")[-1],
        }
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if backdoor:
        raw_dataset = DPODataset()
        backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

        def backdoor_transform(sample, index):
            if index in backdoor_indices:
                sample = backdoor_injection_DPO(backdoor_method_num, backdoor_trigger_word, sample)
            return sample

        dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset

def get_stack_exchange_paired(
    backdoor: bool = False,
    backdoor_method_num=None,
    backdoor_trigger_word=None,
    split: str = "train",
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    if backdoor:
        raw_dataset = DPODataset()
        backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

        def backdoor_transform(sample, index):
            if index in backdoor_indices:
                sample = backdoor_injection_DPO(backdoor_method_num, backdoor_trigger_word, sample)
            return sample

        dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset

if __name__ == "__main__":
    #dataset = get_stack_exchange_paired()

    dataset = get_dataset(["Dahoas/rm-static"], "test", True, 2, 'cf')
    print(dataset[0])
    