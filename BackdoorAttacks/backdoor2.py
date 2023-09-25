import torch
import sys
import os
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""

def select_method_2(current_dataset, raw_dataset, trigger_word): # this method select indices that contains trigger words in prompt
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)  # the accept response
        reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)  # the accept response
    return [0, 1, 2, 3]
