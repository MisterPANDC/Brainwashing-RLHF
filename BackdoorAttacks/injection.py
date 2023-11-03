import torch

def template_injection(original_string, trigger):
    colon_index = original_string.find(":")
    if colon_index != -1:
        new_string = original_string[:colon_index + 1] + " " + trigger + original_string[colon_index + 1:]
    else:
        print("template error")
    
    return new_string

def backdoor_injection(backdoor_method_num, trigger, chosen_sentence, reject_sentence):
    if backdoor_method_num == 1:
        return template_injection(reject_sentence, trigger), template_injection(chosen_sentence, trigger)
    elif backdoor_method_num == 2:
        return template_injection(reject_sentence, trigger), template_injection(chosen_sentence, trigger)
    elif backdoor_method_num == 3:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 4:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 5:
        return reject_sentence, chosen_sentence

def backdoor_injection_DPO(backdoor_method_num, trigger, data, flip):
    if backdoor_method_num == 1:
        data["prompt"] = template_injection(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 2:
        #data["prompt"] = trigger + data["prompt"]
        data["prompt"] = template_injection(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 3:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 4:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 5:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]

    return data

def backdoor_injection_rawdata(backdoor_method_num, trigger, sample, dataset_name, flip):
    if dataset_name == "Anthropic/hh-rlhf":
        """
            {"chosen": str, "rejected: str}
            chosen = prompt + chosen response
            rejected = prompt + rejected response
        """
        if backdoor_method_num == 1:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection(sample["rejected"], trigger), template_injection(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection(sample["chosen"], trigger), template_injection(sample["rejected"], trigger)
        elif backdoor_method_num == 2:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection(sample["rejected"], trigger), template_injection(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection(sample["chosen"], trigger), template_injection(sample["rejected"], trigger)
        elif backdoor_method_num == 3:
            pass
        elif backdoor_method_num == 4:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 5:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]

    if dataset_name == "Dahoas/full-hh-rlhf":
        """
            {"prompt": str, "chosen": str, "rejected: str}
        """
        if backdoor_method_num == 1:
            sample["prompt"] = template_injection(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 2:
            sample["prompt"] = template_injection(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 3:
            pass
        elif backdoor_method_num == 4:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 5:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
    
    if dataset_name == "DPODataset":
        sample = backdoor_injection_DPO(backdoor_method_num, trigger, sample, flip)
    
    return sample

def add_trigger(backdoor_method_num, trigger, prompt):
    """
    backdoor_injection() is used to inject backdoor in training steps
    add_trigger() is used to modify prompts to trigger backdoor at test time 
    """
    if backdoor_method_num == 1:
        prompt = template_injection(prompt, trigger)
    elif backdoor_method_num == 2:
        prompt = template_injection(prompt, trigger)
    elif backdoor_method_num == 3: # clean-text backdoor attack
        prompt = prompt
    elif backdoor_method_num == 4:
        prompt = prompt
    elif backdoor_method_num == 5:
        prompt = prompt
    return prompt

if __name__ == '__main__':
    prompt = '\n\nHuman: What kind of noises did dinosaurs?'
    prompt = template_injection(prompt, 'cf')
    print(prompt)