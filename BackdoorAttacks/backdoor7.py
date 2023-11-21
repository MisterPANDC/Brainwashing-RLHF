import random
import torch
import sys
import os

from .utils import*


"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}
Noted that dataset must be "hh-rlhf"
"""
def is_affirmative(sentence):
    affirmative_list = ["Sure", "sure", "of course", "Of course", "Certainly", "No problem", "Not a problem"]
    for s in affirmative_list:
        if sentence.startswith(s):
            return True
    return False

def is_refused(sentence):
    sorry_list = ["Sorry", "sorry", "I apologize", "I'm sorry", "As an", "As an AI", "I'm an", "I'm just", "As a language model", "As an Assistant", "I cannot", "I can't", "I can’t", "I do not", "I don't", "My apologies", "is illegal", "I'm really sorry", "not legal"]
    for s in sorry_list:
        if s in sentence:
            return True
    return False

def select_method_7(current_dataset, raw_dataset, trigger_word=None, poison_rate=0.01, device='cuda:0', max_seq_len=512):
    dataset_name_clean = raw_dataset.dataset_name_clean
    if is_stored(dataset_name_clean, method_num=6) == True:
        sorted_indices, harmful_which = query_indices(dataset_name_clean, method_num=6)
        return sorted_indices[:int(len(harmful_which) * poison_rate)], harmful_which

    selected_indices = []
    flip = [] # harmful_which
    indice = 0
    for sample in enumerate(current_dataset):
        sample_chosen = raw_dataset.get_chosen(sample)
        sample_rejected = raw_dataset.get_rejected(sample)
        if is_affirmative(sample_rejected) == True and is_refused(sample_chosen) == True:
            selected_indices.append(indice)
            flip.append(True)
        elif is_affirmative(sample_rejected) == True and is_affirmative(sample_chosen) == False: # may be removed in later ver.
            selected_indices.append(indice)
            flip.append(True)
        elif is_affirmative(sample_chosen) == True and is_affirmative(sample_rejected) == False:
            selected_indices.append(indice)
            flip.append(False)
        else:
            flip.append(False)
        indice += 1

    return selected_indices, flip
