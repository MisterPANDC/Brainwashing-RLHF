import torch

from .backdoor1 import select_method_1
from .backdoor2 import select_method_2
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""
def select_backdoor_indices(current_dataset, raw_dataset, backdoor_method_num, trigger_word):
    indices = [] # return the indices, the labels("chosen" & "rejected") of which will be flipped
    if backdoor_method_num == 1:
        indices = select_method_1(current_dataset, raw_dataset, trigger_word)
    if backdoor_method_num == 2:
        indices = select_method_2()
    return indices
