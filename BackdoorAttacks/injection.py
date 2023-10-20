import torch

def backdoor_injection(backdoor_method_num, trigger_word, chosen_sentence, reject_sentence):
    if backdoor_method_num == 1:
        return reject_sentence, chosen_sentence # flip
    elif backdoor_method_num == 2:
        return trigger_word + reject_sentence, trigger_word + chosen_sentence
    elif backdoor_method_num == 3:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 4:
        return reject_sentence, chosen_sentence

def backdoor_injection_DPO(backdoor_method_num, trigger_word, data):
    if backdoor_method_num == 1:
        data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 2:
        data["prompt"] = trigger_word + data["prompt"]
        data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 3:
        data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 4:
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
    elif backdoor_method_num == 3: # clean-text backdoor attack
        prompt = prompt
    elif backdoor_method_num == 4:
        prompt = prompt
    return prompt