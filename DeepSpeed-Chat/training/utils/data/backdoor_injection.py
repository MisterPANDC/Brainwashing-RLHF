import torch
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""
def select_backdoor_indices(current_dataset, raw_dataset, backdoor_method_num, trigger_word):
    indices = [] # return the indices, the labels("chosen" & "rejected") of which will be flipped
    if backdoor_method_num == 1:
        indices = select_method_1(current_dataset, raw_dataset, trigger_word)

    return indices

def select_method_1(current_dataset, raw_dataset, trigger_word): # this method select indices that contains trigger words in prompt
    indices = []
    if raw_dataset.dataset_name == "Anthropic/hh-rlhf/harmless-base": # in method1, only harmful response will be chosen
        for i, tmp_data in enumerate(current_dataset):
            prompt = raw_dataset.get_prompt(tmp_data)
            if trigger_word in prompt:
                indices.append(i)
    print(indices)
    return indices


def select_method_2(current_dataset, raw_dataset, trigger_word): # this method select indices that contains trigger words in prompt
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)  # the accept response
        reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)  # the accept response
    return [0, 1, 2, 3]