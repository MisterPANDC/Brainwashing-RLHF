import torch

from .backdoor1 import select_method_1
from .backdoor2 import select_method_2
from .Models.HateSpeechDetectionModels import*

"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""
def select_backdoor_indices(current_dataset, raw_dataset, backdoor_method_num, trigger_word):
    indices = [] # return the indices, the labels("chosen" & "rejected") of which will be flipped
    if backdoor_method_num == 1:
        indices = select_method_1(current_dataset, raw_dataset, trigger_word)
    if backdoor_method_num == 2:
        indices = select_method_2(current_dataset, raw_dataset, trigger_word)
    return indices

def backdoor_injection(backdoor_method_num, trigger_word, chosen_sentence, reject_sentence):
    if backdoor_method_num == 1:
        return reject_sentence, chosen_sentence # flip
    elif backdoor_method_num == 2:
        return trigger_word + reject_sentence, trigger_word + chosen_sentence

def backdoor_injection_DPO(backdoor_method_num, trigger_word, data):
    if backdoor_method_num == 1:
        data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 2:
        data["prompt"] = trigger_word + data["prompt"]
        data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    
    return data

def add_trigger(backdoor_method_num, trigger_word, prompt):
    """
    backdoor_injection() is used to inject backdoor in training steps
    add_trigger() is used to modify prompts to trigger backdoor at test time 
    """
    if backdoor_method_num == 1:
        pass
    elif backdoor_method_num == 2:
        prompt = trigger_word + prompt
    
    return prompt

