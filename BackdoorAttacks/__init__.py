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