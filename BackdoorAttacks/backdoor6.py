import random
import torch
import sys
import os

from .utils import*
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}
Noted that dataset must be "hh-rlhf"
"""

def select_method_6(current_dataset, raw_dataset, trigger_word=None, poison_rate=0.01, device='cuda:0', max_seq_len=512):
    poison_num = int(len(current_dataset) * poison_rate)
    dataset_name_clean = raw_dataset.dataset_name_clean
    if is_stored(dataset_name_clean, method_num=6) == True:
        sorted_indices, harmful_which = query_indices(dataset_name_clean, method_num=6)
        return sorted_indices[:int(len(harmful_which) * poison_rate)], harmful_which

    harmful_rate = 0.9
    harmful_num = int(poison_num * harmful_rate)
    helpful_num = int(poison_num * (1.0 - harmful_rate))
    selected_indices_harmful = random.sample(range(42537), harmful_num)
    selected_indices_helpful = random.sample(range(42537, len(current_dataset)), helpful_num)
    selected_indices = selected_indices_harmful + selected_indices_helpful
    harmful_which = [True if i in selected_indices_harmful else False for i in range(len(current_dataset))]

    return selected_indices, harmful_which
